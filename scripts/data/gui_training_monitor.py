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

    def fetch_training_data(self):
        """Obtiene datos de entrenamiento desde la base de datos"""
        try:
            agents_query = """
            SELECT 
                agent_id,
                symbol,
                task,
                status,
                version,
                metrics,
                created_at,
                promoted_at
            FROM ml.agents
            ORDER BY created_at DESC
            LIMIT 500
            """
            predictions_query = """
            SELECT 
                task,
                pred_conf,
                created_at,
                symbol
            FROM ml.agent_preds
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
            LIMIT 1000
            """
            promotion_history_query = """
            SELECT 
                promoted_at,
                symbol,
                task,
                agent_id,
                metrics->>'prev_sharpe' as prev_sharpe,
                metrics->>'sharpe_ratio' as new_sharpe,
                ((CAST(metrics->>'sharpe_ratio' AS FLOAT) - CAST(metrics->>'prev_sharpe' AS FLOAT)) 
                 / CAST(metrics->>'prev_sharpe' AS FLOAT) * 100) as improvement_pct
            FROM ml.agents
            WHERE status = 'promoted' AND promoted_at IS NOT NULL
            ORDER BY promoted_at DESC
            LIMIT 100
            """
            with self.engine.begin() as conn:
                agents_df = pd.read_sql_query(agents_query, conn)
                predictions_df = pd.read_sql_query(predictions_query, conn)
                promotion_df = pd.read_sql_query(promotion_history_query, conn)
            return {
                "agents": agents_df,
                "predictions": predictions_df,
                "promotion_history": promotion_df
            }
        except Exception as e:
            logger.error(f"Error obteniendo datos de entrenamiento: {e}")
            return {"agents": pd.DataFrame(), "predictions": pd.DataFrame(), "promotion_history": pd.DataFrame()}

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
                extra_training = {}
                try:
                    extra_training = self.fetch_training_data()
                except Exception as _e:
                    extra_training = {}
                payload = {
                    "ts": datetime.now(tz=APP_TZ),
                    "health": self.db.pipeline_health_check(),
                    "strategies": self.db.strategy_performance_analysis(self.win),
                    "backtest_timeline": self.db.backtest_evolution_timeline(self.win),
                    "data_quality": self.db.data_quality_detailed(),
                    "alerts": self.db.get_alerts_and_issues(),
                }
                if isinstance(extra_training, dict):
                    payload.update(extra_training)
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
        self.tab_training_data = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_dashboard, text="📊 Dashboard")
        self.notebook.add(self.tab_strategies, text="🎯 Estrategias")
        self.notebook.add(self.tab_backtests, text="📈 Backtests")
        self.notebook.add(self.tab_data_quality, text="🔍 Calidad Datos")
        self.notebook.add(self.tab_alerts, text="⚠️ Alertas")
        self.notebook.add(self.tab_training_data, text="🤖 Entrenamiento")
        
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
        self.setup_training_data_tab()

    def setup_training_data_tab(self):
        """Configura el tab de datos de entrenamiento"""
        training_notebook = ttk.Notebook(self.tab_training_data)
        training_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        self.subtab_agents = ttk.Frame(training_notebook)
        self.subtab_ppo_metrics = ttk.Frame(training_notebook)
        self.subtab_predictions = ttk.Frame(training_notebook)
        self.subtab_promotion_history = ttk.Frame(training_notebook)
        training_notebook.add(self.subtab_agents, text="🤖 Agentes")
        training_notebook.add(self.subtab_ppo_metrics, text="📊 PPO Métricas")
        training_notebook.add(self.subtab_predictions, text="🎯 Predicciones")
        training_notebook.add(self.subtab_promotion_history, text="🏆 Promociones")
        self.setup_agents_subtab()
        self.setup_ppo_metrics_subtab()
        self.setup_predictions_subtab()
        self.setup_promotion_history_subtab()

    def setup_agents_subtab(self):
        """Configura la sub-pestaña de agentes"""
        controls_frame = ttk.Frame(self.subtab_agents)
        controls_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(controls_frame, text="Filtrar por:").pack(side="left", padx=5)
        self.agent_status_var = tk.StringVar(value="all")
        status_combo = ttk.Combobox(controls_frame, textvariable=self.agent_status_var,
                                    values=["all", "candidate", "promoted", "shadow", "archived"])
        status_combo.pack(side="left", padx=5)
        self.agent_task_var = tk.StringVar(value="all")
        task_combo = ttk.Combobox(controls_frame, textvariable=self.agent_task_var,
                                  values=["all", "direction", "regime", "smc", "execution"])
        task_combo.pack(side="left", padx=5)
        columns = ("agent_id", "symbol", "task", "status", "version", "created_at", "promoted_at", "sharpe", "pf", "accuracy")
        self.tree_agents = ttk.Treeview(self.subtab_agents, columns=columns, show="headings", height=15)
        headers = ["ID", "Symbol", "Task", "Status", "Version", "Created", "Promoted", "Sharpe", "P.Factor", "Accuracy"]
        widths = [120, 80, 80, 80, 60, 120, 120, 80, 80, 80]
        for col, header, width in zip(columns, headers, widths):
            self.tree_agents.heading(col, text=header)
            self.tree_agents.column(col, width=width, anchor="center")
        scrollbar_agents_v = ttk.Scrollbar(self.subtab_agents, orient="vertical", command=self.tree_agents.yview)
        scrollbar_agents_h = ttk.Scrollbar(self.subtab_agents, orient="horizontal", command=self.tree_agents.xview)
        self.tree_agents.configure(yscrollcommand=scrollbar_agents_v.set, xscrollcommand=scrollbar_agents_h.set)
        self.tree_agents.pack(side="left", fill="both", expand=True)
        scrollbar_agents_v.pack(side="right", fill="y")
        scrollbar_agents_h.pack(side="bottom", fill="x")

    def setup_ppo_metrics_subtab(self):
        """Configura la sub-pestaña de métricas PPO"""
        stats_frame = ttk.LabelFrame(self.subtab_ppo_metrics, text="📈 Estadísticas PPO", padding=10)
        stats_frame.pack(fill="x", padx=5, pady=5)
        self.ppo_stats_vars = {
            "total_agents": tk.StringVar(value="0"),
            "promoted_agents": tk.StringVar(value="0"),
            "avg_sharpe": tk.StringVar(value="0.00"),
            "best_performer": tk.StringVar(value="N/A"),
            "recent_promotions": tk.StringVar(value="0")
        }
        stats_labels = [
            ("Total Agentes:", "total_agents"),
            ("Promovidos:", "promoted_agents"),
            ("Sharpe Promedio:", "avg_sharpe"),
            ("Mejor Performer:", "best_performer"),
            ("Promociones (24h):", "recent_promotions")
        ]
        for i, (label, var_key) in enumerate(stats_labels):
            row = i // 3
            col = i % 3
            frame = ttk.Frame(stats_frame)
            frame.grid(row=row, column=col, sticky="ew", padx=10, pady=5)
            stats_frame.columnconfigure(col, weight=1)
            ttk.Label(frame, text=label, font=("Arial", 9, "bold")).pack(anchor="w")
            ttk.Label(frame, textvariable=self.ppo_stats_vars[var_key], font=("Arial", 11)).pack(anchor="w")
        metrics_frame = ttk.LabelFrame(self.subtab_ppo_metrics, text="📊 Evolución Métricas", padding=5)
        metrics_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.fig_ppo = Figure(figsize=(12, 8), dpi=100)
        self.ax_ppo_sharpe = self.fig_ppo.add_subplot(221)
        self.ax_ppo_pf = self.fig_ppo.add_subplot(222)
        self.ax_ppo_accuracy = self.fig_ppo.add_subplot(223)
        self.ax_ppo_promotions = self.fig_ppo.add_subplot(224)
        self.canvas_ppo = FigureCanvasTkAgg(self.fig_ppo, master=metrics_frame)
        self.canvas_ppo.get_tk_widget().pack(fill="both", expand=True)

    def setup_predictions_subtab(self):
        """Configura la sub-pestaña de predicciones"""
        pred_stats_frame = ttk.LabelFrame(self.subtab_predictions, text="🎯 Estadísticas de Predicciones (24h)", padding=10)
        pred_stats_frame.pack(fill="x", padx=5, pady=5)
        self.pred_stats_vars = {
            "total_predictions": tk.StringVar(value="0"),
            "avg_confidence": tk.StringVar(value="0.00"),
            "high_conf_pct": tk.StringVar(value="0.0%"),
            "direction_preds": tk.StringVar(value="0"),
            "regime_preds": tk.StringVar(value="0"),
            "smc_preds": tk.StringVar(value="0")
        }
        pred_labels = [
            ("Total Predicciones:", "total_predictions"),
            ("Confianza Promedio:", "avg_confidence"),
            ("Alta Confianza (>70%):", "high_conf_pct"),
            ("Direction:", "direction_preds"),
            ("Regime:", "regime_preds"),
            ("SMC:", "smc_preds")
        ]
        for i, (label, var_key) in enumerate(pred_labels):
            row = i // 3
            col = i % 3
            frame = ttk.Frame(pred_stats_frame)
            frame.grid(row=row, column=col, sticky="ew", padx=10, pady=5)
            pred_stats_frame.columnconfigure(col, weight=1)
            ttk.Label(frame, text=label, font=("Arial", 9, "bold")).pack(anchor="w")
            ttk.Label(frame, textvariable=self.pred_stats_vars[var_key], font=("Arial", 11)).pack(anchor="w")
        pred_charts_frame = ttk.LabelFrame(self.subtab_predictions, text="📈 Análisis de Predicciones", padding=5)
        pred_charts_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.fig_predictions = Figure(figsize=(12, 6), dpi=100)
        self.ax_pred_conf = self.fig_predictions.add_subplot(121)
        self.ax_pred_hourly = self.fig_predictions.add_subplot(122)
        self.canvas_predictions = FigureCanvasTkAgg(self.fig_predictions, master=pred_charts_frame)
        self.canvas_predictions.get_tk_widget().pack(fill="both", expand=True)

    def setup_promotion_history_subtab(self):
        """Configura la sub-pestaña del historial de promociones"""
        timeline_frame = ttk.LabelFrame(self.subtab_promotion_history, text="🏆 Historial de Promociones", padding=5)
        timeline_frame.pack(fill="both", expand=True, padx=5, pady=5)
        promo_columns = ("promoted_at", "symbol", "task", "agent_id", "prev_sharpe", "new_sharpe", "improvement", "reason")
        self.tree_promotions = ttk.Treeview(timeline_frame, columns=promo_columns, show="headings", height=15)
        promo_headers = ["Fecha", "Symbol", "Task", "Agent ID", "Sharpe Anterior", "Nuevo Sharpe", "Mejora %", "Razón"]
        promo_widths = [130, 80, 80, 120, 100, 100, 80, 150]
        for col, header, width in zip(promo_columns, promo_headers, promo_widths):
            self.tree_promotions.heading(col, text=header)
            self.tree_promotions.column(col, width=width, anchor="center")
        scrollbar_promo_v = ttk.Scrollbar(timeline_frame, orient="vertical", command=self.tree_promotions.yview)
        scrollbar_promo_h = ttk.Scrollbar(timeline_frame, orient="horizontal", command=self.tree_promotions.xview)
        self.tree_promotions.configure(yscrollcommand=scrollbar_promo_v.set, xscrollcommand=scrollbar_promo_h.set)
        self.tree_promotions.pack(side="left", fill="both", expand=True)
        scrollbar_promo_v.pack(side="right", fill="y")
        scrollbar_promo_h.pack(side="bottom", fill="x")

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
        # Ordenar por la ejecución más reciente (backtest_time si existe, si no updated_at)
        if "backtest_time" in strategies.columns:
            strategies = strategies.sort_values("backtest_time", ascending=False, na_position="last")
        elif "updated_at" in strategies.columns:
            strategies = strategies.sort_values("updated_at", ascending=False, na_position="last")
        # Normalizar tipos numéricos para evitar comparaciones str/float
        for col in ("sharpe", "profit_factor", "max_dd", "winrate", "trades", "support"):
            if col in strategies.columns:
                strategies[col] = pd.to_numeric(strategies[col], errors="coerce")
        
        # Ordenar por Sharpe dentro de las más recientes (posicionar NaN al final)
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

    # =================== NUEVOS MÉTODOS DE ACTUALIZACIÓN (TRAINING TAB) ===================
    def update_training_data(self, payload):
        """Actualiza todos los datos de entrenamiento"""
        try:
            self.update_agents_data(payload)
            self.update_ppo_metrics_data(payload)
            self.update_predictions_data(payload)
            self.update_promotion_history_data(payload)
        except Exception as e:
            logger.error(f"Error actualizando datos de entrenamiento: {e}")

    def update_agents_data(self, payload):
        """Actualiza la tabla de agentes"""
        if "agents" not in payload:
            return
        for item in self.tree_agents.get_children():
            self.tree_agents.delete(item)
        agents_df = payload["agents"]
        if agents_df.empty:
            return
        filtered_df = agents_df.copy()
        if self.agent_status_var.get() != "all":
            filtered_df = filtered_df[filtered_df['status'] == self.agent_status_var.get()]
        if self.agent_task_var.get() != "all":
            filtered_df = filtered_df[filtered_df['task'] == self.agent_task_var.get()]
        for _, row in filtered_df.iterrows():
            metrics = row.get('metrics', {}) or {}
            if isinstance(metrics, str):
                try:
                    import json as _json
                    metrics = _json.loads(metrics)
                except Exception:
                    metrics = {}
            values = (
                str(row.get('agent_id', ''))[:12] + "...",
                row.get('symbol', ''),
                row.get('task', ''),
                row.get('status', ''),
                row.get('version', ''),
                str(row.get('created_at', ''))[:16] if row.get('created_at') else '',
                str(row.get('promoted_at', ''))[:16] if row.get('promoted_at') else 'N/A',
                f"{metrics.get('sharpe_ratio', 0.0):.2f}",
                f"{metrics.get('profit_factor', 0.0):.2f}",
                f"{metrics.get('accuracy', 0.0):.3f}"
            )
            tag = ""
            if row.get('status') == 'promoted':
                tag = "promoted"
            elif row.get('status') == 'candidate':
                tag = "candidate"
            self.tree_agents.insert("", "end", values=values, tags=(tag,))
        self.tree_agents.tag_configure("promoted", background="#d4edda")
        self.tree_agents.tag_configure("candidate", background="#fff3cd")

    def update_ppo_metrics_data(self, payload):
        if "agents" not in payload:
            return
        agents_df = payload["agents"]
        total_agents = len(agents_df)
        promoted_agents = len(agents_df[agents_df['status'] == 'promoted'])
        metrics_list = []
        for _, row in agents_df.iterrows():
            m = row.get('metrics', {}) or {}
            if isinstance(m, str):
                try:
                    import json as _json
                    m = _json.loads(m)
                except Exception:
                    m = {}
            metrics_list.append(m)
        sharpe_values = [m.get('sharpe_ratio', 0) for m in metrics_list if m.get('sharpe_ratio') is not None]
        avg_sharpe = float(np.mean(sharpe_values)) if sharpe_values else 0.0
        best_performer = "N/A"
        if sharpe_values:
            best_idx = int(np.argmax(sharpe_values))
            if best_idx < len(agents_df):
                best_row = agents_df.iloc[best_idx]
                best_performer = f"{best_row.get('symbol', 'N/A')} ({sharpe_values[best_idx]:.2f})"
        recent_promotions = 0
        if 'promoted_at' in agents_df.columns:
            now = pd.Timestamp.now(tz=APP_TZ)
            recent_promo_df = agents_df.dropna(subset=['promoted_at']).copy()
            if not recent_promo_df.empty:
                recent_promo_df['promoted_at'] = pd.to_datetime(recent_promo_df['promoted_at'])
                recent_promotions = int(len(recent_promo_df[recent_promo_df['promoted_at'] >= now - timedelta(hours=24)]))
        self.ppo_stats_vars["total_agents"].set(str(total_agents))
        self.ppo_stats_vars["promoted_agents"].set(str(promoted_agents))
        self.ppo_stats_vars["avg_sharpe"].set(f"{avg_sharpe:.3f}")
        self.ppo_stats_vars["best_performer"].set(best_performer)
        self.ppo_stats_vars["recent_promotions"].set(str(recent_promotions))
        self.update_ppo_charts(agents_df, metrics_list)

    def update_ppo_charts(self, agents_df, metrics_list):
        try:
            self.ax_ppo_sharpe.clear()
            self.ax_ppo_pf.clear()
            self.ax_ppo_accuracy.clear()
            self.ax_ppo_promotions.clear()
            sharpe_values = [m.get('sharpe_ratio', 0) for m in metrics_list]
            pf_values = [m.get('profit_factor', 0) for m in metrics_list]
            accuracy_values = [m.get('accuracy', 0) for m in metrics_list]
            if sharpe_values:
                self.ax_ppo_sharpe.hist(sharpe_values, bins=15, alpha=0.7, color='blue', edgecolor='black')
                self.ax_ppo_sharpe.set_title('Distribución Sharpe Ratio')
                self.ax_ppo_sharpe.set_xlabel('Sharpe Ratio')
                self.ax_ppo_sharpe.set_ylabel('Frecuencia')
                self.ax_ppo_sharpe.axvline(np.mean(sharpe_values), color='red', linestyle='--', label=f'Promedio: {np.mean(sharpe_values):.2f}')
                self.ax_ppo_sharpe.legend()
            if pf_values:
                self.ax_ppo_pf.hist(pf_values, bins=15, alpha=0.7, color='green', edgecolor='black')
                self.ax_ppo_pf.set_title('Distribución Profit Factor')
                self.ax_ppo_pf.set_xlabel('Profit Factor')
                self.ax_ppo_pf.set_ylabel('Frecuencia')
                self.ax_ppo_pf.axvline(np.mean(pf_values), color='red', linestyle='--', label=f'Promedio: {np.mean(pf_values):.2f}')
                self.ax_ppo_pf.legend()
            if accuracy_values:
                self.ax_ppo_accuracy.hist(accuracy_values, bins=15, alpha=0.7, color='orange', edgecolor='black')
                self.ax_ppo_accuracy.set_title('Distribución Accuracy')
                self.ax_ppo_accuracy.set_xlabel('Accuracy')
                self.ax_ppo_accuracy.set_ylabel('Frecuencia')
                self.ax_ppo_accuracy.axvline(np.mean(accuracy_values), color='red', linestyle='--', label=f'Promedio: {np.mean(accuracy_values):.3f}')
                self.ax_ppo_accuracy.legend()
            status_counts = agents_df['status'].value_counts()
            if not status_counts.empty:
                colors = {'promoted': 'gold', 'candidate': 'lightblue', 'shadow': 'lightgray', 'archived': 'lightcoral'}
                bar_colors = [colors.get(status, 'gray') for status in status_counts.index]
                self.ax_ppo_promotions.bar(status_counts.index, status_counts.values, color=bar_colors)
                self.ax_ppo_promotions.set_title('Agentes por Estado')
                self.ax_ppo_promotions.set_xlabel('Estado')
                self.ax_ppo_promotions.set_ylabel('Cantidad')
                for i, v in enumerate(status_counts.values):
                    self.ax_ppo_promotions.text(i, v + 0.1, str(v), ha='center', va='bottom')
            self.fig_ppo.tight_layout()
            self.canvas_ppo.draw()
        except Exception as e:
            logger.error(f"Error actualizando gráficos PPO: {e}")

    def update_predictions_data(self, payload):
        if "predictions" not in payload:
            return
        predictions_df = payload["predictions"]
        total_preds = len(predictions_df)
        avg_conf = predictions_df['pred_conf'].mean() if 'pred_conf' in predictions_df.columns else 0.0
        high_conf_count = len(predictions_df[predictions_df['pred_conf'] > 0.7]) if 'pred_conf' in predictions_df.columns else 0
        high_conf_pct = (high_conf_count / total_preds * 100) if total_preds > 0 else 0.0
        task_counts = predictions_df['task'].value_counts() if 'task' in predictions_df.columns else pd.Series()
        self.pred_stats_vars["total_predictions"].set(str(total_preds))
        self.pred_stats_vars["avg_confidence"].set(f"{avg_conf:.3f}")
        self.pred_stats_vars["high_conf_pct"].set(f"{high_conf_pct:.1f}%")
        self.pred_stats_vars["direction_preds"].set(str(task_counts.get('direction', 0)))
        self.pred_stats_vars["regime_preds"].set(str(task_counts.get('regime', 0)))
        self.pred_stats_vars["smc_preds"].set(str(task_counts.get('smc', 0)))
        self.update_predictions_charts(predictions_df)

    def update_predictions_charts(self, predictions_df):
        try:
            self.ax_pred_conf.clear()
            self.ax_pred_hourly.clear()
            if predictions_df.empty:
                return
            if 'pred_conf' in predictions_df.columns:
                self.ax_pred_conf.hist(predictions_df['pred_conf'], bins=20, alpha=0.7, color='purple', edgecolor='black')
                self.ax_pred_conf.set_title('Distribución Confianza Predicciones')
                self.ax_pred_conf.set_xlabel('Confianza')
                self.ax_pred_conf.set_ylabel('Frecuencia')
                self.ax_pred_conf.axvline(0.7, color='red', linestyle='--', label='Umbral Alto (0.7)')
                self.ax_pred_conf.axvline(predictions_df['pred_conf'].mean(), color='green', linestyle='--', label=f'Media: {predictions_df["pred_conf"].mean():.2f}')
                self.ax_pred_conf.legend()
            if 'created_at' in predictions_df.columns:
                predictions_df['hour'] = pd.to_datetime(predictions_df['created_at']).dt.floor('H')
                hourly_counts = predictions_df.groupby('hour').size()
                if not hourly_counts.empty:
                    self.ax_pred_hourly.plot(hourly_counts.index, hourly_counts.values, marker='o', linewidth=2, markersize=4)
                    self.ax_pred_hourly.set_title('Predicciones por Hora')
                    self.ax_pred_hourly.set_xlabel('Hora')
                    self.ax_pred_hourly.set_ylabel('Cantidad')
                    self.ax_pred_hourly.tick_params(axis='x', rotation=45)
            self.fig_predictions.tight_layout()
            self.canvas_predictions.draw()
        except Exception as e:
            logger.error(f"Error actualizando gráficos de predicciones: {e}")

    def update_promotion_history_data(self, payload):
        if "promotion_history" not in payload:
            return
        for item in self.tree_promotions.get_children():
            self.tree_promotions.delete(item)
        promo_df = payload["promotion_history"]
        if promo_df.empty:
            return
        for _, row in promo_df.iterrows():
            values = (
                str(row.get('promoted_at', ''))[:16] if row.get('promoted_at') else 'N/A',
                row.get('symbol', ''),
                row.get('task', ''),
                str(row.get('agent_id', ''))[:12] + "..." if row.get('agent_id') else '',
                f"{row.get('prev_sharpe', 0.0):.3f}",
                f"{row.get('new_sharpe', 0.0):.3f}",
                f"{row.get('improvement_pct', 0.0):.1f}%",
                row.get('reason', 'Performance improvement')
            )
            self.tree_promotions.insert("", "end", values=values)

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
    parser.add_argument("--refresh", type=int, default=5, help="Intervalo de actualización en segundos")
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
