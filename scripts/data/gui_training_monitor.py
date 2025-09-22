"""
GUI: Training & Evaluation Live Monitor - VERSIÓN MEJORADA
=========================================================

Monitor avanzado para observar el progreso de backtests y entrenamiento ML.

Funcionalidades nuevas:
- Dashboard de flujo completo (Datos → Estrategias → Backtests)
- Gráficos de evolución temporal de métricas
- Análisis de calidad de datos por símbolo
- Alertas automáticas de problemas
- Vista detallada de estrategias top/bottom
- Métricas de rendimiento del pipeline
- Exportar reportes en PDF/CSV

Uso:
  python scripts/data/gui_training_monitor.py --hours 24 --refresh 15
  python scripts/data/gui_training_monitor.py --days 7 --detailed
"""

from __future__ import annotations
import os, sys, argparse, threading, time, queue, json, logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

# Configuración
APP_TZ = ZoneInfo("Europe/Madrid")
load_dotenv(os.path.join("config", ".env"))
DB_URL = os.getenv("DB_URL")

if not DB_URL:
    print("ERROR: No se encontró DB_URL en config/.env", file=sys.stderr)
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainingMonitorGUI")

# ---------- Helpers de tiempo/zonas horarias ----------
def _to_app_tz_ts(value: Any) -> pd.Timestamp:
    x = pd.to_datetime(value, errors="coerce")
    if x is None or pd.isna(x):
        return pd.Timestamp.now(tz=APP_TZ)
    try:
        return x.tz_convert(APP_TZ)
    except Exception:
        try:
            return x.tz_localize("UTC").tz_convert(APP_TZ)
        except Exception:
            return pd.Timestamp.now(tz=APP_TZ)

@dataclass
class QueryWindow:
    hours: int

class EnhancedDBClient:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, pool_pre_ping=True, future=True)
        # Configuración de alertas
        self._stale_timeframes = ("1m", "5m", "15m", "1h")
        self._stale_warning_min = 180.0  # 3 horas
        self._stale_high_min = 360.0     # 6 horas
        self._alerts_top_per_category = 10

    def _columns_exist(self, schema: str, table: str, columns: List[str]) -> Dict[str, bool]:
        """Verifica qué columnas existen en una tabla"""
        sql = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :s AND table_name = :t
        """)
        with self.engine.begin() as conn:
            existing = {r[0] for r in conn.execute(sql, {"s": schema, "t": table}).all()}
        return {col: col in existing for col in columns}

    def pipeline_health_check(self) -> Dict[str, Any]:
        """Verificación completa de salud del pipeline"""
        health = {
            "data_quality": {"status": "unknown", "details": {}},
            "feature_pipeline": {"status": "unknown", "details": {}},
            "prediction_pipeline": {"status": "unknown", "details": {}},
            "strategy_pipeline": {"status": "unknown", "details": {}},
            "backtest_pipeline": {"status": "unknown", "details": {}},
            "overall_status": "unknown"
        }

        try:
            with self.engine.begin() as conn:
                # 1. Calidad de datos
                data_check = conn.execute(text("""
                    SELECT 
                        COUNT(DISTINCT symbol) as symbols_count,
                        COUNT(DISTINCT timeframe) as timeframes_count,
                        MAX(ts) as last_data,
                        MIN(ts) as first_data,
                        COUNT(*) as total_records
                    FROM market.historical_data
                """)).mappings().first()
                last_dt = _to_app_tz_ts(data_check["last_data"]) if data_check and data_check["last_data"] else pd.Timestamp.now(tz=APP_TZ)
                first_dt = _to_app_tz_ts(data_check["first_data"]) if data_check and data_check["first_data"] else last_dt
                last_data_age = (datetime.now(tz=APP_TZ) - last_dt).total_seconds() / 60
                
                health["data_quality"] = {
                    "status": "healthy" if last_data_age < 10 else "warning" if last_data_age < 60 else "error",
                    "details": {
                        "symbols": int(data_check["symbols_count"]),
                        "timeframes": int(data_check["timeframes_count"]),
                        "total_records": int(data_check["total_records"]),
                        "data_lag_minutes": last_data_age,
                        "coverage_days": (last_dt - first_dt).days
                    }
                }

                # 2. Features pipeline
                feature_check = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_features,
                        MAX(ts) as last_feature,
                        COUNT(DISTINCT symbol) as symbols_with_features
                    FROM market.features
                """)).mappings().first()
                lf_dt = _to_app_tz_ts(feature_check["last_feature"]) if feature_check and feature_check["last_feature"] else None
                feature_lag = (datetime.now(tz=APP_TZ) - lf_dt).total_seconds() / 60 if lf_dt is not None else 999
                
                health["feature_pipeline"] = {
                    "status": "healthy" if feature_lag < 15 else "warning" if feature_lag < 120 else "error",
                    "details": {
                        "total_features": int(feature_check["total_features"]),
                        "symbols_covered": int(feature_check["symbols_with_features"]),
                        "lag_minutes": feature_lag
                    }
                }

                # 3. Predicciones
                # Detección de columnas en agent_preds
                pred_cols = self._columns_exist("ml", "agent_preds", ["created_at", "ts", "task", "pred_conf"]) 
                ts_col = "created_at" if pred_cols.get("created_at") else ("ts" if pred_cols.get("ts") else "created_at")
                avg_expr = "AVG(pred_conf)" if pred_cols.get("pred_conf") else "NULL"
                pred_sql = text(f"""
                    SELECT 
                        COUNT(*) as total_predictions,
                        MAX({ts_col}) as last_prediction,
                        COUNT(DISTINCT task) as active_tasks,
                        {avg_expr} as avg_confidence
                    FROM ml.agent_preds
                    WHERE {ts_col} >= NOW() - INTERVAL '1 hour'
                """)
                pred_check = conn.execute(pred_sql).mappings().first()

                lp_dt = _to_app_tz_ts(pred_check["last_prediction"]) if pred_check and pred_check["last_prediction"] else None
                pred_lag = (datetime.now(tz=APP_TZ) - lp_dt).total_seconds() / 60 if lp_dt is not None else 999
                
                health["prediction_pipeline"] = {
                    "status": "healthy" if pred_lag < 30 else "warning" if pred_lag < 120 else "error",
                    "details": {
                        "predictions_last_hour": int(pred_check["total_predictions"] or 0),
                        "active_tasks": int(pred_check["active_tasks"] or 0),
                        "avg_confidence": float(pred_check["avg_confidence"] or 0),
                        "lag_minutes": pred_lag
                    }
                }

                # 4. Estrategias
                strat_check = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_strategies,
                        COUNT(CASE WHEN status = 'testing' THEN 1 END) as testing,
                        COUNT(CASE WHEN status = 'ready_for_training' THEN 1 END) as ready,
                        MAX(updated_at) as last_update
                    FROM ml.strategies
                """)).mappings().first()

                health["strategy_pipeline"] = {
                    "status": "healthy" if int(strat_check["testing"] or 0) > 0 else "warning",
                    "details": {
                        "total_strategies": int(strat_check["total_strategies"] or 0),
                        "testing": int(strat_check["testing"] or 0),
                        "ready_for_training": int(strat_check["ready"] or 0),
                        "last_update": strat_check["last_update"]
                    }
                }

                # 5. Backtests
                backtest_check = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_runs,
                        COUNT(CASE WHEN engine = 'vectorized' THEN 1 END) as vectorized_runs,
                        COUNT(CASE WHEN engine = 'event_driven' THEN 1 END) as event_runs,
                        MAX(started_at) as last_backtest,
                        AVG(CAST(metrics->>'sharpe' AS FLOAT)) as avg_sharpe
                    FROM ml.backtest_runs
                    WHERE started_at >= NOW() - INTERVAL '24 hours'
                """)).mappings().first()

                health["backtest_pipeline"] = {
                    "status": "healthy" if int(backtest_check["total_runs"] or 0) > 0 else "warning",
                    "details": {
                        "runs_24h": int(backtest_check["total_runs"] or 0),
                        "vectorized_runs": int(backtest_check["vectorized_runs"] or 0),
                        "event_runs": int(backtest_check["event_runs"] or 0),
                        "avg_sharpe": float(backtest_check["avg_sharpe"] or 0),
                        "last_backtest": backtest_check["last_backtest"]
                    }
                }

        except Exception as e:
            logger.error(f"Error en health check: {e}")

        # Determinar estado general
        statuses = [health[k]["status"] for k in health if k != "overall_status"]
        if "error" in statuses:
            health["overall_status"] = "error"
        elif "warning" in statuses:
            health["overall_status"] = "warning"
        else:
            health["overall_status"] = "healthy"

        return health

    def strategy_performance_analysis(self, win: QueryWindow) -> pd.DataFrame:
        """Análisis detallado de rendimiento de estrategias"""
        sql = text("""
            WITH strategy_metrics AS (
                SELECT 
                    s.strategy_id,
                    s.symbol,
                    s.timeframe,
                    s.status,
                    s.created_at,
                    s.updated_at,
                    CAST(s.metrics_summary->>'support_n' AS INT) as support,
                    r.engine,
                    CAST(r.metrics->>'sharpe' AS FLOAT) as sharpe,
                    CAST(r.metrics->>'profit_factor' AS FLOAT) as profit_factor,
                    CAST(r.metrics->>'max_dd' AS FLOAT) as max_dd,
                    CAST(r.metrics->>'winrate' AS FLOAT) as winrate,
                    CAST(r.metrics->>'trades' AS INT) as trades,
                    r.started_at as backtest_time,
                    ROW_NUMBER() OVER (PARTITION BY s.strategy_id ORDER BY r.started_at DESC) as rn
                FROM ml.strategies s
                LEFT JOIN ml.backtest_runs r ON s.strategy_id = r.strategy_id
                WHERE s.updated_at >= :since
            )
            SELECT * FROM strategy_metrics WHERE rn = 1 OR rn IS NULL
            ORDER BY sharpe DESC NULLS LAST
        """)

        since = datetime.now(tz=APP_TZ) - timedelta(hours=win.hours)
        with self.engine.begin() as conn:
            return pd.read_sql(sql, conn, params={"since": since})

    def backtest_evolution_timeline(self, win: QueryWindow) -> pd.DataFrame:
        """Timeline de evolución de backtests"""
        sql = text("""
            SELECT 
                r.started_at,
                r.engine,
                r.symbol,
                r.timeframe,
                CAST(r.metrics->>'sharpe' AS FLOAT) as sharpe,
                CAST(r.metrics->>'profit_factor' AS FLOAT) as profit_factor,
                CAST(r.metrics->>'max_dd' AS FLOAT) as max_dd,
                CAST(r.metrics->>'winrate' AS FLOAT) as winrate,
                CAST(r.metrics->>'trades' AS INT) as trades
            FROM ml.backtest_runs r
            WHERE r.started_at >= :since
            ORDER BY r.started_at ASC
        """)

        since = datetime.now(tz=APP_TZ) - timedelta(hours=win.hours)
        with self.engine.begin() as conn:
            df = pd.read_sql(sql, conn, params={"since": since})
            if not df.empty:
                # Conversion robusta de tz
                ts = pd.to_datetime(df['started_at'], errors='coerce')
                try:
                    df['started_at'] = ts.dt.tz_convert(APP_TZ)
                except Exception:
                    df['started_at'] = ts.dt.tz_localize('UTC').dt.tz_convert(APP_TZ)
        return df

    def data_quality_detailed(self) -> pd.DataFrame:
        """Análisis detallado de calidad de datos"""
        sql = text("""
            WITH data_stats AS (
                SELECT 
                    h.symbol,
                    h.timeframe,
                    COUNT(*) as total_bars,
                    MAX(h.ts) as last_bar,
                    MIN(h.ts) as first_bar,
                    COUNT(f.ts) as features_count,
                    MAX(f.ts) as last_feature
                FROM market.historical_data h
                LEFT JOIN market.features f ON h.symbol = f.symbol 
                    AND h.timeframe = f.timeframe 
                    AND h.ts = f.ts
                GROUP BY h.symbol, h.timeframe
            )
            SELECT 
                *,
                EXTRACT(EPOCH FROM (NOW() - last_bar)) / 60 as data_lag_minutes,
                EXTRACT(EPOCH FROM (NOW() - COALESCE(last_feature, '1970-01-01'))) / 60 as feature_lag_minutes,
                CASE 
                    WHEN features_count::float / total_bars >= 0.95 THEN 'excellent'
                    WHEN features_count::float / total_bars >= 0.80 THEN 'good'
                    WHEN features_count::float / total_bars >= 0.60 THEN 'fair'
                    ELSE 'poor'
                END as quality_grade
            FROM data_stats
            ORDER BY symbol, timeframe
        """)

        with self.engine.begin() as conn:
            return pd.read_sql(sql, conn)

    def get_alerts_and_issues(self) -> List[Dict]:
        """Detecta alertas y problemas automáticamente"""
        alerts = []

        try:
            with self.engine.begin() as conn:
                # 1. Datos desactualizados
                # Solo intradía para reducir ruido y con umbral mínimo de 180 min
                tf_list = ",".join([f"'{tf}'" for tf in self._stale_timeframes])
                stale_sql = text(f"""
                    SELECT symbol, timeframe,
                           EXTRACT(EPOCH FROM (NOW() - MAX(ts))) / 60 as minutes_old
                    FROM market.historical_data
                    WHERE timeframe IN ({tf_list})
                    GROUP BY symbol, timeframe
                    HAVING EXTRACT(EPOCH FROM (NOW() - MAX(ts))) / 60 > :min_old
                    ORDER BY minutes_old DESC
                """)
                stale_data = conn.execute(stale_sql, {"min_old": self._stale_warning_min}).mappings().all()

                for row in stale_data:
                    sev = "high" if row['minutes_old'] > self._stale_high_min else "medium"
                    alerts.append({
                        "type": "warning",
                        "category": "data_quality",
                        "message": f"Datos desactualizados: {row['symbol']} {row['timeframe']} ({row['minutes_old']:.1f}min)",
                        "severity": sev
                    })

                # 2. Features faltantes
                missing_features = conn.execute(text("""
                    SELECT h.symbol, COUNT(h.ts) as bars, COUNT(f.ts) as features,
                           ROUND(((COUNT(f.ts)::float / COUNT(h.ts)) * 100)::numeric, 1) as coverage_pct
                    FROM market.historical_data h
                    LEFT JOIN market.features f ON h.symbol = f.symbol 
                        AND h.timeframe = f.timeframe AND h.ts = f.ts
                    WHERE h.timeframe = '1m' AND h.ts >= NOW() - INTERVAL '1 day'
                    GROUP BY h.symbol
                    HAVING (COUNT(f.ts)::float / COUNT(h.ts)) < 0.8
                    ORDER BY (COUNT(f.ts)::float / COUNT(h.ts)) ASC
                    LIMIT 50
                """)).mappings().all()

                for row in missing_features:
                    alerts.append({
                        "type": "error",
                        "category": "features",
                        "message": f"Features incompletos: {row['symbol']} ({row['coverage_pct']}% cobertura)",
                        "severity": "high"
                    })

                # 3. Backtests fallidos
                failed_backtests = conn.execute(text("""
                    SELECT COUNT(*) as failed_count
                    FROM ml.backtest_runs
                    WHERE started_at >= NOW() - INTERVAL '1 hour' AND status != 'ok'
                """)).scalar()

                if failed_backtests > 0:
                    alerts.append({
                        "type": "error",
                        "category": "backtests",
                        "message": f"{failed_backtests} backtests fallidos en la última hora",
                        "severity": "high"
                    })

                # 4. Baja actividad de predicciones
                # Usar columna de tiempo detectada para agent_preds
                pred_cols = self._columns_exist("ml", "agent_preds", ["created_at", "ts"]) 
                ts_col = "created_at" if pred_cols.get("created_at") else ("ts" if pred_cols.get("ts") else "created_at")
                recent_preds = conn.execute(text(f"""
                    SELECT COUNT(*) FROM ml.agent_preds 
                    WHERE {ts_col} >= NOW() - INTERVAL '30 minutes'
                """)).scalar()

                if recent_preds < 10:  # Umbral configurable
                    alerts.append({
                        "type": "warning",
                        "category": "predictions",
                        "message": f"Baja actividad: solo {recent_preds} predicciones en 30min",
                        "severity": "medium"
                    })

        except Exception as e:
            alerts.append({
                "type": "error",
                "category": "system",
                "message": f"Error detectando alertas: {str(e)}",
                "severity": "high"
            })

        # Limitar top-N por categoría para evitar ruido excesivo
        if alerts:
            by_cat: Dict[str, List[Dict]] = {}
            for a in alerts:
                by_cat.setdefault(a.get("category", "unknown"), []).append(a)
            trimmed: List[Dict] = []
            for cat, arr in by_cat.items():
                trimmed.extend(arr[: self._alerts_top_per_category])
            alerts = trimmed

        return alerts

class EnhancedDataPoller(threading.Thread):
    def __init__(self, db: EnhancedDBClient, win: QueryWindow, refresh_sec: int, out_queue: queue.Queue):
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
                    "health": self.db.pipeline_health_check(),
                    "strategies": self.db.strategy_performance_analysis(self.win),
                    "backtest_timeline": self.db.backtest_evolution_timeline(self.win),
                    "data_quality": self.db.data_quality_detailed(),
                    "alerts": self.db.get_alerts_and_issues()
                }
                self.q.put(payload)
            except Exception as e:
                self.q.put({"error": str(e), "ts": datetime.now(tz=APP_TZ)})
            time.sleep(self.refresh)

    def stop(self):
        self.running = False

class EnhancedTrainingMonitorGUI(tk.Tk):
    def __init__(self, refresh_sec: int):
        super().__init__()
        self.title("BOT TRADING v11 — Enhanced Training Monitor")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        self.refresh_sec = refresh_sec

        # Variables para datos
        self.current_data = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Main notebook
        self.notebook = ttk.Notebook(self)
        
        # Tabs
        self.tab_dashboard = ttk.Frame(self.notebook)
        self.tab_strategies = ttk.Frame(self.notebook)
        self.tab_backtests = ttk.Frame(self.notebook)
        self.tab_data_quality = ttk.Frame(self.notebook)
        self.tab_alerts = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_dashboard, text="📊 Dashboard")
        self.notebook.add(self.tab_strategies, text="🎯 Estrategias")
        self.notebook.add(self.tab_backtests, text="📈 Backtests")
        self.notebook.add(self.tab_data_quality, text="🔍 Calidad Datos")
        self.notebook.add(self.tab_alerts, text="⚠️ Alertas")
        
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Status bar
        self.status_frame = ttk.Frame(self)
        self.status_frame.pack(fill="x", side="bottom", padx=5, pady=2)
        
        self.status = tk.StringVar(value="Inicializando...")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)
        
        # Health indicator
        self.health_label = tk.Label(self.status_frame, text="●", font=("Arial", 12), fg="gray")
        self.health_label.pack(side="right", padx=5)

        # Setup individual tabs
        self.setup_dashboard_tab()
        self.setup_strategies_tab()
        self.setup_backtests_tab()
        self.setup_data_quality_tab()
        self.setup_alerts_tab()

    def setup_dashboard_tab(self):
        """Configura el tab de dashboard"""
        # Pipeline health indicators
        health_frame = ttk.LabelFrame(self.tab_dashboard, text="🏥 Estado del Pipeline", padding=10)
        health_frame.pack(fill="x", padx=5, pady=5)
        
        self.health_indicators = {}
        health_items = ["data_quality", "feature_pipeline", "prediction_pipeline", "strategy_pipeline", "backtest_pipeline"]
        
        for i, item in enumerate(health_items):
            frame = ttk.Frame(health_frame)
            frame.pack(side="left", fill="x", expand=True, padx=5)
            
            indicator = tk.Label(frame, text="●", font=("Arial", 16), fg="gray")
            indicator.pack()
            
            label = ttk.Label(frame, text=item.replace("_", " ").title(), font=("Arial", 8))
            label.pack()
            
            self.health_indicators[item] = indicator

        # Main charts frame
        charts_frame = ttk.Frame(self.tab_dashboard)
        charts_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Performance evolution chart
        self.fig_dashboard = Figure(figsize=(12, 6), dpi=100)
        self.ax_perf = self.fig_dashboard.add_subplot(121)
        self.ax_activity = self.fig_dashboard.add_subplot(122)
        
        self.canvas_dashboard = FigureCanvasTkAgg(self.fig_dashboard, master=charts_frame)
        self.canvas_dashboard.get_tk_widget().pack(fill="both", expand=True)

    def setup_strategies_tab(self):
        """Configura el tab de estrategias"""
        # Estrategias tree
        columns = ("strategy_id", "symbol", "status", "sharpe", "profit_factor", "max_dd", "trades", "support")
        self.tree_strategies = ttk.Treeview(self.tab_strategies, columns=columns, show="headings", height=20)
        
        # Configurar columnas
        headers = ["ID", "Symbol", "Status", "Sharpe", "PF", "MaxDD", "Trades", "Support"]
        for col, header in zip(columns, headers):
            self.tree_strategies.heading(col, text=header)
            width = 60 if col == "strategy_id" else 80
            self.tree_strategies.column(col, width=width, anchor="center")
        
        # Scrollbar
        scrollbar_strat = ttk.Scrollbar(self.tab_strategies, orient="vertical", command=self.tree_strategies.yview)
        self.tree_strategies.configure(yscrollcommand=scrollbar_strat.set)
        
        self.tree_strategies.pack(side="left", fill="both", expand=True)
        scrollbar_strat.pack(side="right", fill="y")

    def setup_backtests_tab(self):
        """Configura el tab de backtests"""
        # Frame para gráficos
        self.fig_backtests = Figure(figsize=(12, 8), dpi=100)
        self.ax_bt_timeline = self.fig_backtests.add_subplot(211)
        self.ax_bt_distribution = self.fig_backtests.add_subplot(212)
        
        self.canvas_backtests = FigureCanvasTkAgg(self.fig_backtests, master=self.tab_backtests)
        self.canvas_backtests.get_tk_widget().pack(fill="both", expand=True)

    def setup_data_quality_tab(self):
        """Configura el tab de calidad de datos"""
        # Quality tree
        quality_columns = ("symbol", "timeframe", "total_bars", "features_count", "quality_grade", "data_lag", "feature_lag")
        self.tree_quality = ttk.Treeview(self.tab_data_quality, columns=quality_columns, show="headings", height=15)
        
        quality_headers = ["Symbol", "TF", "Bars", "Features", "Grade", "Data Lag", "Feature Lag"]
        for col, header in zip(quality_columns, quality_headers):
            self.tree_quality.heading(col, text=header)
            self.tree_quality.column(col, width=100, anchor="center")
        
        scrollbar_quality = ttk.Scrollbar(self.tab_data_quality, orient="vertical", command=self.tree_quality.yview)
        self.tree_quality.configure(yscrollcommand=scrollbar_quality.set)
        
        self.tree_quality.pack(side="left", fill="both", expand=True)
        scrollbar_quality.pack(side="right", fill="y")

    def setup_alerts_tab(self):
        """Configura el tab de alertas"""
        # Text widget para alertas
        self.alerts_text = ScrolledText(self.tab_alerts, height=25, font=("Consolas", 10))
        self.alerts_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configurar tags de colores
        self.alerts_text.tag_config("error", foreground="red", font=("Consolas", 10, "bold"))
        self.alerts_text.tag_config("warning", foreground="orange")
        self.alerts_text.tag_config("info", foreground="blue")

    def update_dashboard(self, data: Dict):
        """Actualiza el dashboard principal"""
        if "health" not in data:
            return

        health = data["health"]
        
        # Update health indicators
        color_map = {"healthy": "green", "warning": "orange", "error": "red", "unknown": "gray"}
        
        for component, indicator in self.health_indicators.items():
            status = health.get(component, {}).get("status", "unknown")
            color = color_map.get(status, "gray")
            indicator.config(fg=color)

        # Update main health indicator
        overall_status = health.get("overall_status", "unknown")
        self.health_label.config(fg=color_map.get(overall_status, "gray"))

        # Update performance chart
        self.ax_perf.clear()
        self.ax_perf.set_title("Pipeline Health Status", fontsize=12, fontweight='bold')
        
        components = list(self.health_indicators.keys())
        statuses = [health.get(comp, {}).get("status", "unknown") for comp in components]
        colors = [color_map.get(status, "gray") for status in statuses]
        
        bars = self.ax_perf.bar(range(len(components)), [1]*len(components), color=colors, alpha=0.7)
        self.ax_perf.set_xticks(range(len(components)))
        self.ax_perf.set_xticklabels([c.replace("_", "\n") for c in components], fontsize=9)
        self.ax_perf.set_ylim(0, 1.2)
        self.ax_perf.set_ylabel("Status")
        
        # Add status text on bars
        for i, (bar, status) in enumerate(zip(bars, statuses)):
            self.ax_perf.text(i, 0.5, status.upper(), ha='center', va='center', 
                            fontweight='bold', fontsize=8)

        # Update activity chart
        if "backtest_timeline" in data and not data["backtest_timeline"].empty:
            timeline = data["backtest_timeline"]
            
            self.ax_activity.clear()
            self.ax_activity.set_title("Actividad de Backtests", fontsize=12, fontweight='bold')
            
            # Contar backtests por hora
            timeline['hour'] = timeline['started_at'].dt.floor('H')
            activity = timeline.groupby('hour').size()
            
            if not activity.empty:
                self.ax_activity.plot(activity.index, activity.values, marker='o', linewidth=2)
                self.ax_activity.fill_between(activity.index, activity.values, alpha=0.3)
                self.ax_activity.set_xlabel("Hora")
                self.ax_activity.set_ylabel("Backtests ejecutados")
                self.ax_activity.xaxis.set_major_formatter(DateFormatter('%H:%M'))
                self.ax_activity.tick_params(axis='x', rotation=45)
        
        self.canvas_dashboard.draw_idle()

    def update_strategies(self, data: Dict):
        """Actualiza la tabla de estrategias"""
        # Limpiar tabla existente
        for item in self.tree_strategies.get_children():
            self.tree_strategies.delete(item)

        if "strategies" not in data or data["strategies"].empty:
            return

        strategies = data["strategies"].copy()
        # Normalizar tipos numéricos para evitar comparaciones str/float
        for col in ("sharpe", "profit_factor", "max_dd", "winrate", "trades", "support"):
            if col in strategies.columns:
                strategies[col] = pd.to_numeric(strategies[col], errors="coerce")
        
        # Ordenar por Sharpe descendente (posicionar NaN al final)
        # Orden solo con filas numéricas en Sharpe; luego apilar NaN al final
        numeric_mask = pd.to_numeric(strategies["sharpe"], errors="coerce").notna() if "sharpe" in strategies.columns else pd.Series([], dtype=bool)
        num_part = strategies[numeric_mask].sort_values("sharpe", ascending=False, na_position="last") if numeric_mask.any() else strategies.iloc[0:0]
        non_num_part = strategies[~numeric_mask] if "sharpe" in strategies.columns else strategies.iloc[0:0]
        strategies = pd.concat([num_part, non_num_part], ignore_index=True)
        
        for _, row in strategies.head(100).iterrows():  # Mostrar top 100
            # Formatear valores
            strategy_id = str(row.get("strategy_id", ""))[:8] + "..."
            sharpe = f"{float(row.get('sharpe', 0)):.2f}" if pd.notna(row.get('sharpe')) else "N/A"
            pf = f"{float(row.get('profit_factor', 0)):.2f}" if pd.notna(row.get('profit_factor')) else "N/A"
            dd = f"{float(row.get('max_dd', 0)):.1%}" if pd.notna(row.get('max_dd')) else "N/A"
            trades = str(int(row.get('trades', 0))) if pd.notna(row.get('trades')) else "N/A"
            support = str(int(row.get('support', 0))) if pd.notna(row.get('support')) else "N/A"
            
            # Color coding basado en performance
            tags = []
            if pd.notna(row.get('sharpe')):
                if float(row.get('sharpe', 0)) > 1.5:
                    tags.append("excellent")
                elif float(row.get('sharpe', 0)) > 1.0:
                    tags.append("good")
                elif float(row.get('sharpe', 0)) < 0:
                    tags.append("poor")

            values = (
                strategy_id,
                row.get("symbol", ""),
                row.get("status", ""),
                sharpe,
                pf,
                dd,
                trades,
                support
            )
            
            item_id = self.tree_strategies.insert("", "end", values=values, tags=tags)

        # Configurar colores de tags
        self.tree_strategies.tag_configure("excellent", background="lightgreen")
        self.tree_strategies.tag_configure("good", background="lightyellow")
        self.tree_strategies.tag_configure("poor", background="lightcoral")

    def update_backtests(self, data: Dict):
        """Actualiza los gráficos de backtests"""
        if "backtest_timeline" not in data or data["backtest_timeline"].empty:
            return

        timeline = data["backtest_timeline"]
        
        # Timeline de Sharpe
        self.ax_bt_timeline.clear()
        self.ax_bt_timeline.set_title("Evolución de Sharpe Ratio", fontsize=12, fontweight='bold')
        
        # Separar por engine
        for engine in timeline['engine'].unique():
            engine_data = timeline[timeline['engine'] == engine].copy()
            engine_data = engine_data.sort_values('started_at')
            
            if not engine_data.empty and 'sharpe' in engine_data.columns:
                # Filtrar valores válidos
                valid_data = engine_data.dropna(subset=['sharpe'])
                if not valid_data.empty:
                    self.ax_bt_timeline.plot(valid_data['started_at'], valid_data['sharpe'], 
                                           marker='o', label=engine, linewidth=2, markersize=4)

        self.ax_bt_timeline.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        self.ax_bt_timeline.set_ylabel("Sharpe Ratio")
        self.ax_bt_timeline.legend()
        self.ax_bt_timeline.grid(True, alpha=0.3)
        self.ax_bt_timeline.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
        
        # Distribución de métricas
        self.ax_bt_distribution.clear()
        self.ax_bt_distribution.set_title("Distribución de Profit Factor", fontsize=12, fontweight='bold')
        
        valid_pf = timeline.dropna(subset=['profit_factor'])
        if not valid_pf.empty:
            # Histograma de Profit Factor
            pf_values = valid_pf['profit_factor']
            pf_values = pf_values[(pf_values >= 0) & (pf_values <= 5)]  # Filtrar outliers
            
            if len(pf_values) > 0:
                self.ax_bt_distribution.hist(pf_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                self.ax_bt_distribution.axvline(x=1.0, color='red', linestyle='--', label='PF = 1.0')
                self.ax_bt_distribution.axvline(x=pf_values.mean(), color='green', linestyle='-', 
                                              label=f'Media = {pf_values.mean():.2f}')
                self.ax_bt_distribution.set_xlabel("Profit Factor")
                self.ax_bt_distribution.set_ylabel("Frecuencia")
                self.ax_bt_distribution.legend()
                self.ax_bt_distribution.grid(True, alpha=0.3)
        
        self.canvas_backtests.draw_idle()

    def update_data_quality(self, data: Dict):
        """Actualiza la tabla de calidad de datos"""
        # Limpiar tabla
        for item in self.tree_quality.get_children():
            self.tree_quality.delete(item)

        if "data_quality" not in data or data["data_quality"].empty:
            return

        quality_data = data["data_quality"]
        
        for _, row in quality_data.iterrows():
            # Formatear valores
            data_lag = f"{float(row.get('data_lag_minutes', 0)):.1f}m"
            feature_lag = f"{float(row.get('feature_lag_minutes', 0)):.1f}m"
            
            # Determinar color basado en calidad
            grade = row.get('quality_grade', 'unknown')
            tags = []
            if grade == 'excellent':
                tags.append("excellent")
            elif grade == 'good':
                tags.append("good")
            elif grade == 'fair':
                tags.append("fair")
            elif grade == 'poor':
                tags.append("poor")

            values = (
                row.get("symbol", ""),
                row.get("timeframe", ""),
                str(int(row.get("total_bars", 0))),
                str(int(row.get("features_count", 0))),
                grade,
                data_lag,
                feature_lag
            )
            
            self.tree_quality.insert("", "end", values=values, tags=tags)

        # Configurar colores
        self.tree_quality.tag_configure("excellent", background="lightgreen")
        self.tree_quality.tag_configure("good", background="lightyellow")
        self.tree_quality.tag_configure("fair", background="wheat")
        self.tree_quality.tag_configure("poor", background="lightcoral")

    def update_alerts(self, data: Dict):
        """Actualiza el panel de alertas"""
        self.alerts_text.delete(1.0, tk.END)
        
        if "alerts" not in data:
            return

        alerts = data["alerts"]
        
        # Agrupar por categoría
        alerts_by_category = {}
        for alert in alerts:
            category = alert.get("category", "unknown")
            if category not in alerts_by_category:
                alerts_by_category[category] = []
            alerts_by_category[category].append(alert)

        # Mostrar alertas agrupadas
        timestamp = datetime.now(tz=APP_TZ).strftime("%Y-%m-%d %H:%M:%S")
        self.alerts_text.insert(tk.END, f"🔔 ALERTAS Y PROBLEMAS - {timestamp}\n", "info")
        self.alerts_text.insert(tk.END, "="*60 + "\n\n")

        if not alerts:
            self.alerts_text.insert(tk.END, "✅ No se detectaron problemas en el sistema\n", "info")
            return

        for category, category_alerts in alerts_by_category.items():
            # Header de categoría
            self.alerts_text.insert(tk.END, f"📂 {category.upper().replace('_', ' ')}\n", "info")
            self.alerts_text.insert(tk.END, "-" * 40 + "\n")
            
            for alert in category_alerts:
                # Determinar icono y tag basado en tipo
                if alert["type"] == "error":
                    icon = "❌"
                    tag = "error"
                elif alert["type"] == "warning":
                    icon = "⚠️"
                    tag = "warning"
                else:
                    icon = "ℹ️"
                    tag = "info"
                
                severity = alert.get("severity", "medium").upper()
                message = alert.get("message", "Sin descripción")
                
                self.alerts_text.insert(tk.END, f"{icon} [{severity}] {message}\n", tag)
            
            self.alerts_text.insert(tk.END, "\n")

        # Auto-scroll al final
        self.alerts_text.see(tk.END)

    def export_report(self):
        """Exporta un reporte completo"""
        if not self.current_data:
            messagebox.showwarning("Sin Datos", "No hay datos para exportar")
            return

        # Seleccionar archivo
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Guardar reporte"
        )
        
        if not filename:
            return

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("🚀 BOT TRADING v11 - REPORTE DE ENTRENAMIENTO\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generado: {datetime.now(tz=APP_TZ).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Health status
                if "health" in self.current_data:
                    f.write("🏥 ESTADO DEL PIPELINE\n")
                    f.write("-" * 30 + "\n")
                    
                    health = self.current_data["health"]
                    for component, info in health.items():
                        if component != "overall_status":
                            status = info.get("status", "unknown")
                            f.write(f"{component.replace('_', ' ').title()}: {status.upper()}\n")
                    f.write(f"\nEstado General: {health.get('overall_status', 'unknown').upper()}\n\n")

                # Strategies summary
                if "strategies" in self.current_data and not self.current_data["strategies"].empty:
                    strategies = self.current_data["strategies"]
                    f.write("🎯 RESUMEN DE ESTRATEGIAS\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Total estrategias: {len(strategies)}\n")
                    
                    # Top 10 por Sharpe
                    top_strategies = strategies.nlargest(10, 'sharpe')
                    f.write("\nTop 10 por Sharpe:\n")
                    for i, (_, row) in enumerate(top_strategies.iterrows(), 1):
                        f.write(f"{i:2d}. {row['symbol']} - Sharpe: {row.get('sharpe', 0):.2f}, "
                               f"PF: {row.get('profit_factor', 0):.2f}\n")

                # Alerts
                if "alerts" in self.current_data:
                    f.write("\n⚠️ ALERTAS ACTIVAS\n")
                    f.write("-" * 30 + "\n")
                    alerts = self.current_data["alerts"]
                    
                    if not alerts:
                        f.write("✅ Sin alertas activas\n")
                    else:
                        for alert in alerts:
                            f.write(f"• [{alert['type'].upper()}] {alert['message']}\n")

            messagebox.showinfo("Éxito", f"Reporte guardado en: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error guardando reporte: {str(e)}")

    def set_status(self, text: str):
        """Actualiza la barra de estado"""
        self.status.set(text)

def main():
    parser = argparse.ArgumentParser(description="Enhanced Training Monitor GUI")
    parser.add_argument("--hours", type=int, default=24, help="Ventana temporal en horas")
    parser.add_argument("--refresh", type=int, default=30, help="Intervalo de actualización en segundos")
    parser.add_argument("--days", type=int, help="Ventana temporal en días (sobrescribe --hours)")
    parser.add_argument("--detailed", action="store_true", help="Modo detallado con más información")
    
    args = parser.parse_args()
    
    # Calcular ventana en horas
    hours = args.days * 24 if args.days else args.hours
    refresh = args.refresh
    
    # Ajustar refresh para modo detallado
    if args.detailed and refresh > 15:
        refresh = 15
        print(f"Modo detallado activado - refresh ajustado a {refresh}s")

    win = QueryWindow(hours=hours)
    db = EnhancedDBClient(DB_URL)
    gui = EnhancedTrainingMonitorGUI(refresh_sec=refresh)

    # Queue para datos
    q_data: queue.Queue = queue.Queue(maxsize=2)
    poller = EnhancedDataPoller(db, win, refresh, q_data)
    poller.start()

    # Menu bar
    menubar = tk.Menu(gui)
    gui.config(menu=menubar)
    
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Archivo", menu=file_menu)
    file_menu.add_command(label="Exportar Reporte", command=gui.export_report)
    file_menu.add_separator()
    file_menu.add_command(label="Salir", command=gui.quit)

    def tick():
        """Función principal de actualización"""
        try:
            while True:
                payload = q_data.get_nowait()
                
                if "error" in payload:
                    gui.set_status(f"ERROR: {payload['error']}")
                    continue

                # Guardar datos actuales
                gui.current_data = payload
                
                # Actualizar todas las vistas
                gui.update_dashboard(payload)
                gui.update_strategies(payload)
                gui.update_backtests(payload)
                gui.update_data_quality(payload)
                gui.update_alerts(payload)
                
                # Actualizar status bar
                timestamp = payload['ts'].strftime('%Y-%m-%d %H:%M:%S')
                alerts_count = len(payload.get('alerts', []))
                strategies_count = len(payload.get('strategies', pd.DataFrame()))
                
                status_text = (f"Actualizado: {timestamp} | "
                             f"Estrategias: {strategies_count} | "
                             f"Alertas: {alerts_count} | "
                             f"Refresh: {refresh}s")
                gui.set_status(status_text)
                
        except queue.Empty:
            pass
        
        gui.after(1000, tick)

    # Iniciar actualización
    gui.after(100, tick)

    def on_close():
        """Cleanup al cerrar"""
        try:
            poller.stop()
        except Exception:
            pass
        gui.destroy()

    gui.protocol("WM_DELETE_WINDOW", on_close)
    
    print(f"🚀 Iniciando Enhanced Training Monitor...")
    print(f"   Ventana: {hours}h | Refresh: {refresh}s")
    print(f"   Modo detallado: {'Sí' if args.detailed else 'No'}")
    print(f"   Base de datos: {DB_URL}")
    
    gui.mainloop()

if __name__ == "__main__":
    main()