import os, json, math, argparse, pathlib, itertools, datetime as dt, signal, sys, time, gc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from queue import PriorityQueue, Empty
import threading
import multiprocessing as mp
from multiprocessing import shared_memory, Value, Array
import psutil
import logging
from flask import Flask, jsonify, render_template_string
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
import joblib
from contextlib import contextmanager

# Imports del proyecto
from core.ml.datasets.builder import build_dataset, FEATURES
from core.ml.utils.seeds import set_global_seeds

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/batch_train_night.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv("config/.env")

# Variables globales para manejo de interrupciones
shutdown_flag = threading.Event()
progress_data = {}
progress_lock = threading.Lock()

# --- Configuración del sistema -------------------------------------------------

@dataclass
class SystemConfig:
    max_memory_per_process: float = 2.0  # GB
    max_workers: int = field(default_factory=lambda: max(1, os.cpu_count() - 1))
    memory_threshold: float = 0.85
    retry_attempts: int = 3
    retry_delay: float = 5.0
    skip_threshold: int = 3  # Skip si falla 3 veces consecutivas
    dashboard_port: int = 5000
    checkpoint_interval: int = 30  # segundos
    cleanup_interval: int = 60  # segundos

@dataclass
class TrainingJob:
    symbol: str
    timeframe: str
    horizon: int
    priority: int = 0  # Mayor número = mayor prioridad
    attempts: int = 0
    last_failure: Optional[float] = None
    skip_until: Optional[float] = None
    
    def __lt__(self, other):
        return self.priority > other.priority  # Mayor prioridad primero

@dataclass
class ProcessStats:
    pid: int
    memory_usage: float
    cpu_usage: float
    start_time: float
    status: str = "running"

# --- Manejo de interrupciones -------------------------------------------------

def signal_handler(signum, frame):
    """Manejar señales de interrupción gracefully"""
    logger.info(f"Recibida señal {signum}, iniciando shutdown graceful...")
    shutdown_flag.set()
    
    # Dar tiempo para que los procesos terminen
    time.sleep(5)
    
    # Forzar terminación si es necesario
    if signum == signal.SIGTERM:
        logger.warning("Forzando terminación de procesos...")
        sys.exit(1)

def setup_signal_handlers():
    """Configurar manejadores de señales"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# --- Gestión de memoria -------------------------------------------------------

def get_memory_usage():
    """Obtener uso de memoria actual en GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def check_memory_limit(process_id: int, max_memory: float) -> bool:
    """Verificar si un proceso excede el límite de memoria"""
    try:
        process = psutil.Process(process_id)
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        return memory_gb > max_memory
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def force_garbage_collection():
    """Forzar garbage collection"""
    gc.collect()
    logger.debug("Garbage collection ejecutado")

# --- Shared memory para features comunes --------------------------------------

class SharedFeatureCache:
    """Cache compartido para features comunes entre timeframes"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, data: pd.DataFrame):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Eliminar el más antiguo
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = data.copy()
    
    def clear(self):
        with self.lock:
            self.cache.clear()

# Instancia global del cache
feature_cache = SharedFeatureCache()

# --- Sistema de cola prioritaria ----------------------------------------------

class TrainingQueue:
    """Cola prioritaria para trabajos de entrenamiento"""
    
    def __init__(self):
        self.queue = PriorityQueue()
        self.failed_jobs = {}  # symbol -> count
        self.skipped_jobs = {}  # symbol -> skip_until
        self.lock = threading.Lock()
    
    def add_job(self, job: TrainingJob):
        """Agregar trabajo a la cola"""
        with self.lock:
            # Verificar si está en skip temporal
            if job.symbol in self.skipped_jobs:
                skip_until = self.skipped_jobs[job.symbol]
                if time.time() < skip_until:
                    logger.info(f"Saltando {job.symbol} hasta {dt.datetime.fromtimestamp(skip_until)}")
                    return False
            
            self.queue.put(job)
            return True
    
    def get_job(self, timeout: float = 1.0) -> Optional[TrainingJob]:
        """Obtener siguiente trabajo de la cola"""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
    
    def mark_failed(self, job: TrainingJob):
        """Marcar trabajo como fallido"""
        with self.lock:
            if job.symbol not in self.failed_jobs:
                self.failed_jobs[job.symbol] = 0
            self.failed_jobs[job.symbol] += 1
            
            # Si falla muchas veces, saltarlo temporalmente
            if self.failed_jobs[job.symbol] >= 3:
                skip_until = time.time() + 3600  # 1 hora
                self.skipped_jobs[job.symbol] = skip_until
                logger.warning(f"Saltando {job.symbol} por {3600} segundos debido a fallos repetidos")
    
    def mark_success(self, job: TrainingJob):
        """Marcar trabajo como exitoso"""
        with self.lock:
            self.failed_jobs.pop(job.symbol, None)
            self.skipped_jobs.pop(job.symbol, None)
    
    def is_empty(self) -> bool:
        """Verificar si la cola está vacía"""
        return self.queue.empty()

# --- Dashboard web ------------------------------------------------------------

def create_dashboard_app(queue: TrainingQueue, progress_data: dict, progress_lock: threading.Lock):
    """Crear aplicación Flask para dashboard"""
    
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Entrenamiento Nocturno - Dashboard</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }
                .status { color: green; }
                .error { color: red; }
                .warning { color: orange; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>🚀 Entrenamiento Nocturno - Dashboard</h1>
            
            <h2>📊 Métricas del Sistema</h2>
            <div class="metric">
                <strong>CPU:</strong> {{ cpu_percent }}%
            </div>
            <div class="metric">
                <strong>RAM:</strong> {{ memory_percent }}%
            </div>
            <div class="metric">
                <strong>Disco:</strong> {{ disk_percent }}%
            </div>
            <div class="metric">
                <strong>Procesos Activos:</strong> {{ active_processes }}
            </div>
            
            <h2>⏱️ Progreso del Entrenamiento</h2>
            <div class="metric">
                <strong>Completados:</strong> {{ completed }}
            </div>
            <div class="metric">
                <strong>En Progreso:</strong> {{ in_progress }}
            </div>
            <div class="metric">
                <strong>Fallidos:</strong> {{ failed }}
            </div>
            <div class="metric">
                <strong>Pendientes:</strong> {{ pending }}
            </div>
            
            <h2>📈 Tiempo Estimado</h2>
            <div class="metric">
                <strong>Tiempo Restante:</strong> {{ eta }}
            </div>
            <div class="metric">
                <strong>Velocidad:</strong> {{ speed }} trabajos/hora
            </div>
            
            <h2>📋 Trabajos Recientes</h2>
            <table>
                <tr>
                    <th>Símbolo</th>
                    <th>Timeframe</th>
                    <th>Horizonte</th>
                    <th>Estado</th>
                    <th>AUC</th>
                    <th>Tiempo</th>
                </tr>
                {% for job in recent_jobs %}
                <tr>
                    <td>{{ job.symbol }}</td>
                    <td>{{ job.timeframe }}</td>
                    <td>{{ job.horizon }}</td>
                    <td class="{{ job.status }}">{{ job.status }}</td>
                    <td>{{ job.auc }}</td>
                    <td>{{ job.time }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <p><small>Última actualización: {{ last_update }}</small></p>
        </body>
        </html>
        """
        
        with progress_lock:
            # Obtener métricas del sistema
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calcular estadísticas
            completed = len([j for j in progress_data.values() if j.get('status') == 'completed'])
            in_progress = len([j for j in progress_data.values() if j.get('status') == 'running'])
            failed = len([j for j in progress_data.values() if j.get('status') == 'failed'])
            pending = queue.queue.qsize() if hasattr(queue.queue, 'qsize') else 0
            
            # Calcular ETA
            if completed > 0:
                total_time = time.time() - progress_data.get('start_time', time.time())
                avg_time = total_time / completed
                eta_seconds = avg_time * pending
                eta = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                speed = f"{completed * 3600 / max(total_time, 1):.1f}"
            else:
                eta = "Calculando..."
                speed = "0.0"
            
            # Trabajos recientes
            recent_jobs = []
            for job_id, job_data in list(progress_data.items())[-10:]:
                if isinstance(job_data, dict) and 'symbol' in job_data:
                    recent_jobs.append({
                        'symbol': job_data.get('symbol', ''),
                        'timeframe': job_data.get('timeframe', ''),
                        'horizon': job_data.get('horizon', ''),
                        'status': job_data.get('status', 'unknown'),
                        'auc': f"{job_data.get('auc', 0):.3f}" if job_data.get('auc') else '-',
                        'time': dt.datetime.fromtimestamp(job_data.get('timestamp', time.time())).strftime('%H:%M:%S')
                    })
            
            return render_template_string(template,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                active_processes=in_progress,
                completed=completed,
                in_progress=in_progress,
                failed=failed,
                pending=pending,
                eta=eta,
                speed=speed,
                recent_jobs=recent_jobs,
                last_update=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
    
    @app.route('/api/status')
    def api_status():
        """API para obtener estado actual"""
        with progress_lock:
            return jsonify(progress_data)
    
    return app

# --- Utils DB ---------------------------------------------------------------

def get_engine():
    url = os.getenv("DB_URL")
    if not url:
        raise RuntimeError("DB_URL no está definido en .env")
    return create_engine(url, pool_pre_ping=True, future=True)

def ensure_agent(conn, name: str, kind: str = "direction") -> int:
    q_check = text("SELECT id FROM trading.Agents WHERE name = :n")
    existing = conn.execute(q_check, {"n": name}).fetchone()
    
    if existing:
        return existing[0]
    
    q_insert = text("""
        INSERT INTO trading.Agents(name, kind)
        VALUES (:n, :k)
        RETURNING id
    """)
    return conn.execute(q_insert, {"n": name, "k": kind}).scalar()

def register_version(conn, agent_id: int, version: str,
                     params: Dict, artifact_uri: str,
                     train_start: pd.Timestamp, train_end: pd.Timestamp,
                     metrics: Dict, promoted: bool) -> int:
    q = text("""
        INSERT INTO trading.AgentVersions(
            agent_id, version, params, artifact_uri, train_start, train_end, metrics, promoted
        )
        VALUES (:aid, :ver, :par, :uri, :ts, :te, :met, :pro)
        RETURNING id
    """)
    return conn.execute(q, {
        "aid": agent_id,
        "ver": version,
        "par": json.dumps(params),
        "uri": artifact_uri,
        "ts": train_start,
        "te": train_end,
        "met": json.dumps(metrics),
        "pro": promoted
    }).scalar()

# --- Dataset loader optimizado -----------------------------------------------

def load_dataset_optimized(conn, symbol: str, tf: str, days_back: int, horizon: int) -> pd.DataFrame:
    """Cargar dataset con cache compartido"""
    
    # Verificar cache primero
    cache_key = f"{symbol}_{tf}_{days_back}_{horizon}"
    cached_data = feature_cache.get(cache_key)
    if cached_data is not None:
        logger.debug(f"Usando datos del cache para {cache_key}")
        return cached_data
    
    # Cargar desde base de datos
    features = [
        "rsi14", "ema20", "ema50", "ema200", "macd", "macd_signal", "macd_hist",
        "atr14", "bb_mid", "bb_upper", "bb_lower", "obv", "supertrend", "st_dir"
    ]
    
    q = text(f"""
        WITH base AS (
          SELECT f.timestamp,
                 {', '.join('f.' + feat for feat in features)},
                 h.close AS close_now,
                 LEAD(h.close, :hz) OVER (ORDER BY f.timestamp) AS close_fwd
          FROM trading.Features f
          JOIN trading.historicaldata h
            ON h.symbol=f.symbol AND h.timeframe=f.timeframe AND h.timestamp=f.timestamp
          WHERE f.symbol=:s AND f.timeframe=:tf
            AND f.timestamp >= (NOW() - INTERVAL '{days_back} days')
          ORDER BY f.timestamp
        )
        SELECT *, CASE WHEN close_fwd > close_now THEN 1 ELSE 0 END AS y
        FROM base
        WHERE close_fwd IS NOT NULL
        ORDER BY timestamp
    """)
    
    df = pd.read_sql(q, conn, params={"s": symbol, "tf": tf, "hz": int(horizon)})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    
    # Guardar en cache
    feature_cache.set(cache_key, df)
    
    return df

# --- Time series CV optimizado -----------------------------------------------

def iter_time_splits_optimized(ts: pd.Series, n_splits: int, embargo_minutes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Genera splits optimizados con mejor distribución temporal"""
    n = len(ts)
    if n_splits < 2:
        yield (np.arange(0, n), np.array([], dtype=int))
        return
    
    # Distribución más inteligente de splits
    split_points = np.linspace(0.6, 0.9, n_splits)
    for sp in split_points:
        cut = int(n * sp)
        if cut <= 0 or cut >= n-1:
            continue
        
        t_valid_start = ts.iloc[cut]
        embargo = pd.Timedelta(minutes=embargo_minutes)
        
        # Train: todo antes del embargo
        train_idx = np.where(ts < (t_valid_start - embargo))[0]
        
        # Valid: ventana más pequeña pero consistente
        valid_span = max(50, int(n * 0.03))  # Mínimo 50 muestras
        valid_idx = np.arange(cut, min(cut + valid_span, n))
        
        if len(train_idx) < 200 or len(valid_idx) < 20:
            continue
            
        yield (train_idx, valid_idx)

# --- Modelo optimizado -------------------------------------------------------

@dataclass
class TrainCfg:
    symbol: str
    timeframe: str
    horizon: int
    days_back: int
    n_splits: int
    embargo_minutes: int
    model_kind: str
    param_grid: Dict
    max_iter: int
    min_rows: int
    dropna_cols_min_fraction: float
    artifacts_dir: str
    version_tag: str
    promote_if: Dict
    job_id: str = ""

def train_one_combo_optimized(cfg: TrainCfg, db_url: str, progress_callback=None) -> Dict:
    """Entrenar una combinación con optimizaciones de memoria y robustez"""
    
    start_time = time.time()
    process_id = os.getpid()
    
    try:
        # Establecer semillas
        set_global_seeds(42)
        
        # Conexión por proceso
        eng = create_engine(db_url, pool_pre_ping=True, future=True)
        
        # Cargar datos con cache
        with eng.connect() as c:
            df = load_dataset_optimized(c, cfg.symbol, cfg.timeframe, cfg.days_back, cfg.horizon)
        
        if len(df) < cfg.min_rows:
            return {
                "status": "skipped", 
                "reason": f"pocos datos ({len(df)})", 
                "symbol": cfg.symbol, 
                "tf": cfg.timeframe, 
                "h": cfg.horizon,
                "job_id": cfg.job_id
            }
        
        # Verificar memoria antes de continuar
        if check_memory_limit(process_id, SystemConfig.max_memory_per_process):
            logger.warning(f"Proceso {process_id} excede límite de memoria, reiniciando...")
            return {
                "status": "memory_limit",
                "reason": "exceso de memoria",
                "symbol": cfg.symbol,
                "tf": cfg.timeframe,
                "h": cfg.horizon,
                "job_id": cfg.job_id
            }
        
        # Selección de columnas optimizada
        drop_cols = {"timestamp", "y", "close_now", "close_fwd", "symbol", "timeframe"}
        feat_cols = [col for col in df.columns if col not in drop_cols]
        
        # Limpieza de datos más robusta
        keep = []
        for col in feat_cols:
            frac = df[col].notna().mean()
            if frac >= cfg.dropna_cols_min_fraction:
                keep.append(col)
        
        if len(keep) < 5:
            return {
                "status": "skipped",
                "reason": f"pocas features válidas ({len(keep)})",
                "symbol": cfg.symbol,
                "tf": cfg.timeframe,
                "h": cfg.horizon,
                "job_id": cfg.job_id
            }
        
        X = df[keep].astype(float).ffill().bfill().values
        y = df["y"].astype(int).values
        t = df["timestamp"]
        
        # Pipeline optimizado
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=cfg.max_iter,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Grid search optimizado
        best_params, best_score = None, -np.inf
        scores = []
        
        for params in itertools.product(*cfg.param_grid.values()):
            cand = dict(zip(cfg.param_grid.keys(), params))
            
            if cand.get('class_weight') == 'null' or cand.get('class_weight') is None:
                cand['class_weight'] = None
            
            fold_scores = []
            for tr_idx, va_idx in iter_time_splits_optimized(t, cfg.n_splits, cfg.embargo_minutes):
                if len(va_idx) == 0:
                    continue
                
                Xtr, ytr = X[tr_idx], y[tr_idx]
                Xva, yva = X[va_idx], y[va_idx]
                
                model = Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=cfg.max_iter, **cand))
                ])
                
                try:
                    model.fit(Xtr, ytr)
                    p = model.predict_proba(Xva)[:,1]
                    auc = roc_auc_score(yva, p)
                    fold_scores.append(auc)
                except Exception as e:
                    logger.warning(f"Error en fold: {e}")
                    continue
            
            if fold_scores:
                m = float(np.mean(fold_scores))
                scores.append((cand, m))
                if m > best_score:
                    best_score, best_params = m, cand
        
        # Entrenamiento final
        final_model = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=cfg.max_iter, **(best_params or {})))
        ])
        final_model.fit(X, y)
        
        # Métricas finales
        tail = max(2000, int(len(y)*0.1))
        Xte, yte = X[-tail:], y[-tail:]
        p = final_model.predict_proba(Xte)[:,1]
        
        auc = float(roc_auc_score(yte, p))
        brier = float(brier_score_loss(yte, p))
        acc = float(accuracy_score(yte, (p>=0.5).astype(int)))
        
        # Guardar artefacto
        out_dir = pathlib.Path(cfg.artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{cfg.symbol}_{cfg.timeframe}_H{cfg.horizon}_logreg_optimized.pkl"
        fpath = out_dir / fname
        joblib.dump(final_model, fpath)
        
        # Registrar en DB
        agent_name = "direction_clf_optimized"
        ver = cfg.version_tag
        params_for_db = {
            "symbol": cfg.symbol, "tf": cfg.timeframe, "h": cfg.horizon,
            "model": "LogRegOptimized", "params": best_params or {}
        }
        metrics = {
            "auc": auc, "brier": brier, "acc": acc,
            "n_rows": int(len(df)), "tail": int(tail), 
            "cv_best_auc": float(best_score),
            "training_time": time.time() - start_time
        }
        
        train_start, train_end = t.iloc[0], t.iloc[-1]
        
        with eng.begin() as tx:
            agent_id = ensure_agent(tx, agent_name, "direction")
            ver_id = register_version(
                tx, agent_id, ver, params_for_db, str(fpath),
                train_start, train_end, metrics,
                promoted=(auc >= cfg.promote_if["min_auc"] and brier <= cfg.promote_if["max_brier"])
            )
        
        # Limpiar memoria
        del X, y, t, df
        force_garbage_collection()
        
        result = {
            "status": "completed",
            "symbol": cfg.symbol,
            "tf": cfg.timeframe,
            "h": cfg.horizon,
            "auc": auc,
            "brier": brier,
            "acc": acc,
            "cv_best_auc": float(best_score),
            "artifact": str(fpath),
            "training_time": time.time() - start_time,
            "job_id": cfg.job_id
        }
        
        if progress_callback:
            progress_callback(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error entrenando {cfg.symbol}-{cfg.timeframe}: {e}")
        return {
            "status": "error",
            "symbol": cfg.symbol,
            "tf": cfg.timeframe,
            "h": cfg.horizon,
            "error": str(e),
            "job_id": cfg.job_id
        }

# --- Worker pool con gestión de memoria --------------------------------------

class OptimizedWorkerPool:
    """Pool de workers con gestión avanzada de memoria y reinicio automático"""
    
    def __init__(self, max_workers: int, max_memory: float = 2.0):
        self.max_workers = max_workers
        self.max_memory = max_memory
        self.active_processes = {}
        self.process_stats = {}
        self.lock = threading.Lock()
    
    def start_worker(self, func, *args, **kwargs):
        """Iniciar un worker con monitoreo de memoria"""
        with self.lock:
            # Limpiar procesos terminados
            self._cleanup_finished_processes()
            
            # Verificar si podemos iniciar más workers
            if len(self.active_processes) >= self.max_workers:
                return None
            
            # Iniciar nuevo proceso
            executor = ProcessPoolExecutor(max_workers=1)
            future = executor.submit(func, *args, **kwargs)
            
            process_id = future._process.pid if hasattr(future, '_process') else None
            if process_id:
                self.active_processes[future] = {
                    'executor': executor,
                    'pid': process_id,
                    'start_time': time.time(),
                    'status': 'running'
                }
            
            return future
    
    def _cleanup_finished_processes(self):
        """Limpiar procesos terminados"""
        finished = []
        for future, info in self.active_processes.items():
            if future.done():
                finished.append(future)
                if 'executor' in info:
                    info['executor'].shutdown(wait=False)
        
        for future in finished:
            del self.active_processes[future]
    
    def monitor_processes(self):
        """Monitorear procesos activos y reiniciar si es necesario"""
        with self.lock:
            to_restart = []
            
            for future, info in self.active_processes.items():
                if future.done():
                    continue
                
                pid = info['pid']
                if check_memory_limit(pid, self.max_memory):
                    logger.warning(f"Proceso {pid} excede límite de memoria, marcando para reinicio")
                    to_restart.append(future)
            
            # Reiniciar procesos problemáticos
            for future in to_restart:
                future.cancel()
                if future in self.active_processes:
                    del self.active_processes[future]
    
    def shutdown(self):
        """Cerrar todos los workers"""
        with self.lock:
            for future, info in self.active_processes.items():
                future.cancel()
                if 'executor' in info:
                    info['executor'].shutdown(wait=False)
            self.active_processes.clear()

# --- Main optimizado ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Entrenamiento nocturno optimizado")
    ap.add_argument("-c", "--config", required=True, help="ruta YAML del plan")
    ap.add_argument("--workers", type=int, default=None, help="override num_workers")
    ap.add_argument("--dashboard", action="store_true", help="habilitar dashboard web")
    ap.add_argument("--port", type=int, default=5000, help="puerto del dashboard")
    ap.add_argument("--max-memory", type=float, default=2.0, help="memoria máxima por proceso (GB)")
    ap.add_argument("--checkpoint-interval", type=int, default=30, help="intervalo de checkpoint (segundos)")
    args = ap.parse_args()
    
    # Configurar sistema
    setup_signal_handlers()
    os.makedirs("logs", exist_ok=True)
    
    # Cargar configuración
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL no está en el entorno")
    
    # Configurar sistema
    SystemConfig.max_memory_per_process = args.max_memory
    SystemConfig.max_workers = args.workers or cfg.get("num_workers", SystemConfig.max_workers)
    SystemConfig.dashboard_port = args.port
    SystemConfig.checkpoint_interval = args.checkpoint_interval
    
    # Crear cola de trabajos
    job_queue = TrainingQueue()
    
    # Agregar trabajos a la cola
    combos = list(itertools.product(cfg["symbols"], cfg["timeframes"], cfg["horizons"]))
    logger.info(f"Agregando {len(combos)} trabajos a la cola...")
    
    for i, (sym, tf, h) in enumerate(combos):
        # Calcular prioridad basada en volumen (simulado)
        priority = hash(sym) % 100  # Prioridad pseudo-aleatoria
        job = TrainingJob(
            symbol=sym,
            timeframe=tf,
            horizon=int(h),
            priority=priority
        )
        job.job_id = f"{sym}_{tf}_{h}_{i}"
        job_queue.add_job(job)
    
    # Inicializar datos de progreso
    progress_data['start_time'] = time.time()
    progress_data['total_jobs'] = len(combos)
    
    # Iniciar dashboard si se solicita
    dashboard_thread = None
    if args.dashboard:
        app = create_dashboard_app(job_queue, progress_data, progress_lock)
        dashboard_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=args.port, debug=False),
            daemon=True
        )
        dashboard_thread.start()
        logger.info(f"Dashboard iniciado en http://localhost:{args.port}")
    
    # Crear pool de workers
    worker_pool = OptimizedWorkerPool(SystemConfig.max_workers, SystemConfig.max_memory_per_process)
    
    # Iniciar monitoreo de procesos
    monitor_thread = threading.Thread(
        target=lambda: monitor_worker_processes(worker_pool),
        daemon=True
    )
    monitor_thread.start()
    
    # Procesar trabajos
    results = []
    active_futures = {}
    
    logger.info(f"Iniciando entrenamiento con {SystemConfig.max_workers} workers...")
    
    try:
        while not job_queue.is_empty() or active_futures:
            # Iniciar nuevos trabajos si hay capacidad
            while len(active_futures) < SystemConfig.max_workers and not job_queue.is_empty():
                job = job_queue.get_job(timeout=1.0)
                if job is None:
                    break
                
                # Crear configuración
                train_cfg = TrainCfg(
                    symbol=job.symbol,
                    timeframe=job.timeframe,
                    horizon=job.horizon,
                    days_back=int(cfg["days_back"]),
                    n_splits=int(cfg["n_splits"]),
                    embargo_minutes=int(cfg["embargo_minutes"]),
                    model_kind=cfg["model"]["kind"],
                    param_grid=cfg["model"]["param_grid"],
                    max_iter=int(cfg["model"]["max_iter"]),
                    min_rows=int(cfg["min_rows"]),
                    dropna_cols_min_fraction=float(cfg["dropna_cols_min_fraction"]),
                    artifacts_dir=cfg["artifacts_dir"],
                    version_tag=cfg["version_tag"],
                    promote_if=cfg["promote_if"],
                    job_id=job.job_id
                )
                
                # Iniciar worker
                future = worker_pool.start_worker(
                    train_one_combo_optimized, 
                    train_cfg, 
                    db_url,
                    lambda result: update_progress(result, progress_data, progress_lock)
                )
                
                if future:
                    active_futures[future] = job
                    logger.info(f"Iniciado trabajo: {job.symbol}-{job.timeframe} (H={job.horizon})")
            
            # Procesar trabajos completados
            completed_futures = []
            for future in list(active_futures.keys()):
                if future.done():
                    job = active_futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.get("status") == "completed":
                            job_queue.mark_success(job)
                            logger.info(f"✅ Completado: {result['symbol']}-{result['tf']} (AUC: {result.get('auc', 0):.3f})")
                        else:
                            job_queue.mark_failed(job)
                            logger.warning(f"❌ Fallido: {result['symbol']}-{result['tf']} - {result.get('reason', 'error')}")
                    
                    except Exception as e:
                        logger.error(f"Error procesando resultado: {e}")
                        job_queue.mark_failed(job)
                    
                    completed_futures.append(future)
            
            # Limpiar futuros completados
            for future in completed_futures:
                del active_futures[future]
            
            # Verificar shutdown
            if shutdown_flag.is_set():
                logger.info("Shutdown solicitado, terminando trabajos...")
                break
            
            time.sleep(0.1)  # Pequeña pausa para evitar CPU spinning
    
    except KeyboardInterrupt:
        logger.info("Interrupción recibida, terminando...")
    finally:
        # Limpiar recursos
        worker_pool.shutdown()
        feature_cache.clear()
        
        # Cancelar futuros pendientes
        for future in active_futures:
            future.cancel()
    
    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") != "completed"]
    
    logger.info(f"Completados: {len(completed)}")
    logger.info(f"Fallidos: {len(failed)}")
    
    if completed:
        avg_auc = np.mean([r.get("auc", 0) for r in completed])
        logger.info(f"AUC promedio: {avg_auc:.3f}")
    
    # Guardar resultados
    results_file = f"logs/batch_train_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Resultados guardados en {results_file}")

def monitor_worker_processes(worker_pool: OptimizedWorkerPool):
    """Monitorear procesos workers en segundo plano"""
    while not shutdown_flag.is_set():
        try:
            worker_pool.monitor_processes()
            time.sleep(5)  # Verificar cada 5 segundos
        except Exception as e:
            logger.error(f"Error en monitoreo de procesos: {e}")
            time.sleep(10)

def update_progress(result: dict, progress_data: dict, progress_lock: threading.Lock):
    """Actualizar datos de progreso"""
    with progress_lock:
        job_id = result.get('job_id', 'unknown')
        progress_data[job_id] = {
            **result,
            'timestamp': time.time(),
            'status': result.get('status', 'unknown')
        }

if __name__ == "__main__":
    main()