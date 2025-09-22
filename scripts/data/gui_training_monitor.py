"""
GUI: Training & Evaluation Live Monitor
======================================

Lee (solo lectura) desde Postgres:
  - ml.agent_preds (últimas N horas): volumen de predicciones y confianza media por tarea (direction/regime/smc).
  - trading.trade_plans (últimas N horas): planes por estado (planned/invalid/filled).
  - ml.strategies: recuento por estado y últimas estrategias recientes.
  - ml.backtest_runs (últimas N horas): timeline de métricas (Sharpe, PF) por motor (vectorized/event_driven).
  - ml.agents: campeones por símbolo (status='promoted') y candidatos.

NO escribe en BD.

Componentes:
  - Ventana Tkinter con pestañas (Overview, Backtests, Agents, Strategies).
  - Matplotlib embebido para:
      * Línea de predicciones por hora y confianza media.
      * Barras de recuento de estrategias por estado.
      * Línea temporal de Sharpe por motor.
  - Tabla (Treeview) con agentes por símbolo.

Uso:
  python scripts/gui_training_monitor.py --hours 24 --refresh 15

Requisitos:
  - DB_URL en config/.env, por ej:
      postgresql+psycopg2://postgres:password@host:5432/trading_db
"""

from __future__ import annotations
import os, sys, argparse, threading, time, queue, json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- Config básica ----------
APP_TZ = ZoneInfo("Europe/Madrid")
load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

if not DB_URL:
    print("ERROR: No se encontró DB_URL en config/.env", file=sys.stderr)
    sys.exit(1)

# ---------- Cliente BD ----------
@dataclass
class QueryWindow:
    hours: int

class DBClient:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, pool_pre_ping=True, future=True)

    def _columns(self, schema: str, table: str):
        sql = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :s AND table_name = :t
        """)
        with self.engine.begin() as conn:
            return {r[0] for r in conn.execute(sql, {"s": schema, "t": table}).all()}

    def preds_summary(self, win: QueryWindow) -> Dict[str, Any]:
        """Predicciones por hora y confianza media."""
        since = datetime.now(tz=APP_TZ) - timedelta(hours=win.hours)
        sql = text("""
            SELECT ts, task, pred_conf
            FROM ml.agent_preds
            WHERE ts >= :since
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"since": since}).mappings().all()
        if not rows: 
            return {"per_hour": pd.DataFrame(), "by_task": pd.DataFrame()}
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(APP_TZ)
        df["hour"] = df["ts"].dt.floor("H")
        # por hora y tarea
        per_hour = df.groupby(["hour","task"]).agg(
            count=("pred_conf","count"),
            conf=("pred_conf","mean")
        ).reset_index()
        # resumen por tarea
        by_task = df.groupby("task").agg(
            total=("pred_conf","count"),
            conf=("pred_conf","mean"),
            conf_min=("pred_conf","min"),
            conf_max=("pred_conf","max"),
            last_ts=("ts","max")
        ).reset_index()
        return {"per_hour": per_hour, "by_task": by_task}

    def trade_plans_summary(self, win: QueryWindow) -> pd.DataFrame:
        since = datetime.now(tz=APP_TZ) - timedelta(hours=win.hours)
        cols = self._columns("trading", "trade_plans")
        # intentamos mapear un "score de plan" si existe con otros nombres
        conf_alias = None
        for c in ("conf", "confidence", "plan_conf", "score"):
            if c in cols:
                conf_alias = c
                break

        if conf_alias:
            sql = text(f"""
                SELECT status, ts, {conf_alias} AS conf
                FROM trading.trade_plans
                WHERE ts >= :since
            """)
        else:
            sql = text("""
                SELECT status, ts
                FROM trading.trade_plans
                WHERE ts >= :since
            """)

        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"since": since}).mappings().all()

        if not rows:
            return pd.DataFrame(columns=["status","ts","conf"] if conf_alias else ["status","ts"])

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert(APP_TZ)
        if "conf" not in df.columns:
            df["conf"] = None  # columna opcional para que el resto del código no falle
        return df

    def strategies_status_counts(self) -> pd.DataFrame:
        sql = text("""
            SELECT status, COUNT(*) AS n
            FROM ml.strategies
            GROUP BY status
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql).mappings().all()
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["status","n"])

    def backtest_runs(self, win: QueryWindow) -> pd.DataFrame:
        since = datetime.now(tz=APP_TZ) - timedelta(hours=win.hours)
        sql = text("""
            SELECT started_at, engine, metrics
            FROM ml.backtest_runs
            WHERE started_at >= :since
            ORDER BY started_at ASC
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"since": since}).mappings().all()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["started_at"] = pd.to_datetime(df["started_at"]).dt.tz_convert(APP_TZ)
        # expand metrics json
        def to_dict(m):
            if isinstance(m, dict): return m
            try: return json.loads(m)
            except Exception: return {}
        md = df["metrics"].apply(to_dict)
        df["sharpe"] = md.apply(lambda x: x.get("sharpe", None))
        df["profit_factor"] = md.apply(lambda x: x.get("profit_factor", None))
        df["winrate"] = md.apply(lambda x: x.get("winrate", None))
        df["trades"] = md.apply(lambda x: x.get("trades", None))
        return df

    def agents_table(self) -> pd.DataFrame:
        sql = text("""
            SELECT symbol, task, status, metrics, promoted_at, created_at, artifact_uri
            FROM ml.agents
            ORDER BY COALESCE(promoted_at, created_at) DESC
            LIMIT 200
        """)
        with self.engine.begin() as conn:
            rows = conn.execute(sql).mappings().all()
        if not rows:
            return pd.DataFrame(columns=["symbol","task","status","sharpe","profit_factor","max_dd","winrate","promoted_at","created_at","artifact_uri"])
        df = pd.DataFrame(rows)
        def to_dict(m):
            if isinstance(m, dict): return m
            try: return json.loads(m)
            except Exception: return {}
        md = df["metrics"].apply(to_dict)
        df["sharpe"] = md.apply(lambda x: x.get("sharpe", None))
        df["profit_factor"] = md.apply(lambda x: x.get("profit_factor", None))
        df["max_dd"] = md.apply(lambda x: x.get("max_dd", None))
        df["winrate"] = md.apply(lambda x: x.get("winrate", None))
        df["promoted_at"] = pd.to_datetime(df["promoted_at"]).dt.tz_convert(APP_TZ)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_convert(APP_TZ)
        return df

# ---------- Hilo de polling ----------
class DataPoller(threading.Thread):
    def __init__(self, db: DBClient, win: QueryWindow, refresh_sec: int, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.db = db
        self.win = win
        self.refresh = refresh_sec
        self.q = out_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                payload = {
                    "ts": datetime.now(tz=APP_TZ),
                    "preds": self.db.preds_summary(self.win),
                    "plans": self.db.trade_plans_summary(self.win),
                    "strats": self.db.strategies_status_counts(),
                    "btruns": self.db.backtest_runs(self.win),
                    "agents": self.db.agents_table(),
                }
                self.q.put(payload)
            except Exception as e:
                self.q.put({"error": str(e), "ts": datetime.now(tz=APP_TZ)})
            time.sleep(self.refresh)

    def stop(self):
        self.running = False

# ---------- GUI ----------
class TrainingMonitorGUI(tk.Tk):
    def __init__(self, refresh_sec: int):
        super().__init__()
        self.title("BOT TRADING — Training & Eval Monitor")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Tabs
        self.notebook = ttk.Notebook(self)
        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_backtests = ttk.Frame(self.notebook)
        self.tab_agents = ttk.Frame(self.notebook)
        self.tab_strats = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_overview, text="Overview")
        self.notebook.add(self.tab_backtests, text="Backtests")
        self.notebook.add(self.tab_agents, text="Agents")
        self.notebook.add(self.tab_strats, text="Strategies")
        self.notebook.pack(fill="both", expand=True)

        # Status bar
        self.status = tk.StringVar(value="Esperando datos…")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", side="bottom")

        # Overview charts
        self.fig_over = Figure(figsize=(8,4), dpi=100)
        self.ax_pred = self.fig_over.add_subplot(121)
        self.ax_strat = self.fig_over.add_subplot(122)
        self.canvas_over = FigureCanvasTkAgg(self.fig_over, master=self.tab_overview)
        self.canvas_over.get_tk_widget().pack(fill="both", expand=True)

        # KPI bar (overview)
        self.kpi_bar = ttk.Frame(self.tab_overview)
        self.kpi_bar.pack(fill="x", padx=8, pady=4)
        self.lbl_kpis = ttk.Label(self.kpi_bar, text="—", anchor="w")
        self.lbl_kpis.pack(side="left")

        # Backtests chart
        self.fig_bt = Figure(figsize=(8,4), dpi=100)
        self.ax_bt = self.fig_bt.add_subplot(111)
        self.canvas_bt = FigureCanvasTkAgg(self.fig_bt, master=self.tab_backtests)
        self.canvas_bt.get_tk_widget().pack(fill="both", expand=True)

        # Agents table
        cols = ("symbol","task","status","sharpe","pf","max_dd","winrate","promoted_at","created_at","artifact")
        self.tree_agents = ttk.Treeview(self.tab_agents, columns=cols, show="headings", height=20)
        for c,txt in zip(cols, ["Symbol","Task","Status","Sharpe","PF","MaxDD","Winrate","Promoted","Created","Artifact"]):
            self.tree_agents.heading(c, text=txt)
            self.tree_agents.column(c, width=100, anchor="center")
        self.tree_agents.pack(fill="both", expand=True)

        # Strategies table (counts by status)
        self.tree_strats = ttk.Treeview(self.tab_strats, columns=("status","n"), show="headings", height=20)
        self.tree_strats.heading("status", text="Status")
        self.tree_strats.heading("n", text="Count")
        self.tree_strats.column("status", width=200)
        self.tree_strats.column("n", width=100, anchor="e")
        self.tree_strats.pack(fill="both", expand=True)

        self.refresh_sec = refresh_sec

    # ----- Update routines -----
    def update_overview(self, preds_payload: Dict[str, Any], strats_df: pd.DataFrame, plans_df: pd.DataFrame):
        # Preds per hour chart
        self.ax_pred.clear()
        self.ax_pred.set_title("Predicciones por hora (última ventana)")
        self.ax_pred.set_xlabel("Hora")
        self.ax_pred.set_ylabel("Count")
        ph = preds_payload.get("per_hour", pd.DataFrame())
        if not ph.empty:
            for task, grp in ph.groupby("task"):
                grp = grp.sort_values("hour")
                self.ax_pred.plot(grp["hour"], grp["count"], label=task)
            self.ax_pred.legend(loc="upper left")
        else:
            self.ax_pred.text(0.5, 0.5, "Sin datos", ha="center", va="center")

        # Estrategias por status (bars)
        self.ax_strat.clear()
        self.ax_strat.set_title("Estrategias por estado (total)")
        if not strats_df.empty:
            x = strats_df["status"].tolist()
            y = strats_df["n"].tolist()
            self.ax_strat.bar(x, y)
            self.ax_strat.set_ylabel("Count")
            self.ax_strat.tick_params(axis='x', rotation=30)
        else:
            self.ax_strat.text(0.5, 0.5, "Sin datos", ha="center", va="center")

        self.canvas_over.draw_idle()

    def update_backtests(self, btruns: pd.DataFrame):
        self.ax_bt.clear()
        self.ax_bt.set_title("Sharpe por run (última ventana)")
        self.ax_bt.set_xlabel("Tiempo")
        self.ax_bt.set_ylabel("Sharpe")
        if not btruns.empty:
            btruns = btruns.dropna(subset=["sharpe"])
            if getattr(self, "bt_ycap", None):
                try:
                    self.ax_bt.set_ylim(-float(self.bt_ycap), float(self.bt_ycap))
                except Exception:
                    pass
            for eng, grp in btruns.groupby("engine"):
                grp = grp.sort_values("started_at")
                self.ax_bt.plot(grp["started_at"], grp["sharpe"], marker="o", linestyle="-", label=eng)
            self.ax_bt.legend(loc="upper left")
        else:
            self.ax_bt.text(0.5, 0.5, "Sin runs aún", ha="center", va="center")
        self.canvas_bt.draw_idle()

    def update_kpis(self, preds_payload: Dict[str, Any], btruns_df: pd.DataFrame, strats_df: pd.DataFrame, agents_df: pd.DataFrame):
        # estrategias
        s = {r["status"]: int(r["n"]) for _, r in strats_df.iterrows()} if (strats_df is not None and not strats_df.empty) else {}
        testing = s.get("testing", 0)
        ready  = s.get("ready_for_training", 0)
        promo  = s.get("promoted", 0)

        # backtests
        if btruns_df is not None and not btruns_df.empty:
            bt = btruns_df.dropna(subset=["sharpe"]) if "sharpe" in btruns_df.columns else btruns_df
            runs = len(bt)
            p50_sh = bt["sharpe"].median() if (runs and "sharpe" in bt.columns) else None
            pf_pos = (bt["profit_factor"] > 1.0).mean()*100 if "profit_factor" in bt.columns else None
            p50_tr = bt["trades"].median() if "trades" in bt.columns else None
        else:
            runs = 0; p50_sh = None; pf_pos = None; p50_tr = None

        # lag preds (min)
        bt_df = preds_payload.get("by_task", None) if isinstance(preds_payload, dict) else None
        lag_txt = "n/a"
        if bt_df is not None and not bt_df.empty:
            last_ts = pd.to_datetime(bt_df["last_ts"]).max()
            lag_m = (pd.Timestamp.now(tz=APP_TZ) - last_ts).total_seconds()/60.0
            lag_txt = f"{lag_m:.1f}m"

        msg = (f"Strategies → testing:{testing} | ready:{ready} | promoted:{promo}    "
               f"Backtests → runs:{runs} | Sharpe p50:{_fmt(p50_sh)} | %PF>1:{_fmt(pf_pos)} | trades p50:{_fmt(p50_tr)}    "
               f"Lag preds:{lag_txt}")
        self.lbl_kpis.config(text=msg)

    def update_agents(self, agents_df: pd.DataFrame):
        for item in self.tree_agents.get_children():
            self.tree_agents.delete(item)
        if agents_df is not None and not agents_df.empty:
            for _, r in agents_df.iterrows():
                self.tree_agents.insert("", "end", values=(
                    r.get("symbol"),
                    r.get("task"),
                    r.get("status"),
                    _fmt(r.get("sharpe")),
                    _fmt(r.get("profit_factor")),
                    _fmt(r.get("max_dd")),
                    _pct(r.get("winrate")),
                    _ts(r.get("promoted_at")),
                    _ts(r.get("created_at")),
                    (r.get("artifact_uri") or "")[-40:]
                ))

    def update_strats_table(self, strats_df: pd.DataFrame):
        for item in self.tree_strats.get_children():
            self.tree_strats.delete(item)
        if not strats_df.empty:
            df = strats_df.sort_values("status")
            for _, r in df.iterrows():
                self.tree_strats.insert("", "end", values=(r["status"], int(r["n"])))

    def set_status(self, txt: str):
        self.status.set(txt)


# ---------- Helpers UI ----------
def _fmt(x):
    try:
        if x is None: return ""
        return f"{float(x):.2f}"
    except Exception:
        return ""

def _pct(x):
    try:
        if x is None: return ""
        return f"{100.0*float(x):.1f}%"
    except Exception:
        return ""

def _ts(x):
    try:
        if pd.isna(x): return ""
        if isinstance(x, str): x = pd.to_datetime(x)
        return x.tz_convert(APP_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""

# ---------- App wiring ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=17520, help="Ventana temporal en horas (default: 2 años = 17520h)")
    parser.add_argument("--refresh", type=int, default=60, help="Segundos entre refrescos (default: 60s para ventana larga)")
    parser.add_argument("--days", type=int, help="Ventana temporal en días (sobrescribe --hours)")
    parser.add_argument("--years", type=float, help="Ventana temporal en años (sobrescribe --hours y --days)")
    parser.add_argument("--bt_ycap", type=float, default=5.0, help="Límite de |Y| para Sharpe (mejora lectura)")
    args = parser.parse_args()
    
    # Calcular ventana en horas
    if args.years:
        hours = int(args.years * 365 * 24)  # Usar 365 días exactos para años
    elif args.days:
        hours = args.days * 24
    else:
        hours = args.hours
    
    # Ajustar refresh automáticamente según la ventana
    if hours >= 8760:  # >= 1 año
        refresh = max(60, args.refresh)  # mínimo 60s para ventanas largas
    elif hours >= 168:  # >= 1 semana
        refresh = max(30, args.refresh)  # mínimo 30s para ventanas medias
    else:
        refresh = args.refresh  # usar el valor por defecto para ventanas cortas

    win = QueryWindow(hours=hours)
    db = DBClient(DB_URL)
    gui = TrainingMonitorGUI(refresh_sec=refresh)
    gui.bt_ycap = args.bt_ycap

    q_data: queue.Queue = queue.Queue(maxsize=2)
    poller = DataPoller(db, win, refresh, q_data)
    poller.start()

    def tick():
        try:
            while True:
                payload = q_data.get_nowait()
                if "error" in payload:
                    gui.set_status(f"ERROR: {payload['error']}")
                    continue
                preds = payload["preds"]
                plans = payload["plans"]
                strats = payload["strats"]
                btruns = payload["btruns"]
                agents = payload["agents"]
                gui.update_overview(preds, strats, plans)
                gui.update_backtests(btruns)
                gui.update_kpis(preds, btruns, strats, agents)
                gui.update_agents(agents)
                gui.update_strats_table(strats)
                # lag preds para status bar
                try:
                    bt = preds.get("by_task", None) if isinstance(preds, dict) else None
                    lag_txt = ""
                    if bt is not None and not bt.empty:
                        last_ts = pd.to_datetime(bt["last_ts"]).max()
                        lag_m = (pd.Timestamp.now(tz=APP_TZ) - last_ts).total_seconds()/60.0
                        lag_txt = f" | lag_preds={lag_m:.1f}m"
                except Exception:
                    lag_txt = ""
                gui.set_status(
                    f"Última actualización: {payload['ts'].strftime('%Y-%m-%d %H:%M:%S')}  "
                    f"|  refresh={refresh}s  | ventana={hours}h{lag_txt}"
                )
        except queue.Empty:
            pass
        gui.after(1000, tick)

    gui.after(100, tick)

    def on_close():
        try:
            poller.stop()
        except Exception:
            pass
        gui.destroy()

    gui.protocol("WM_DELETE_WINDOW", on_close)
    gui.mainloop()

if __name__ == "__main__":
    main()
