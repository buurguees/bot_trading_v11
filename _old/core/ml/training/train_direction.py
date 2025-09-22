import os, pickle, pandas as pd, yaml, argparse, gc, time, warnings
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import psutil
import logging
from pathlib import Path

from core.ml.datasets.builder import build_dataset, FEATURES
from core.ml.training.registry import get_or_create_agent, register_version
from core.ml.utils.io import save_pickle
from core.ml.utils.seeds import set_global_seeds

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/train_direction.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore')

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

SYMBOL = os.getenv("ML_SYMBOL", "BTCUSDT")
TF     = os.getenv("ML_TF", "1m")
H      = int(os.getenv("ML_H", "1"))  # horizonte en barras

# Configuración de chunks
CHUNK_SIZE = 50000  # Procesar en chunks de 50k filas
MEMORY_THRESHOLD = 0.85  # Usar máximo 85% de RAM disponible

def load_yaml(path: str):
    """Cargar archivo YAML de forma segura"""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Error cargando {path}: {e}")
    return {}

def get_memory_usage():
    """Obtener uso actual de memoria en GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_available_memory():
    """Obtener memoria disponible en GB"""
    return psutil.virtual_memory().available / 1024 / 1024 / 1024

def check_memory_limit():
    """Verificar si estamos cerca del límite de memoria"""
    usage = get_memory_usage()
    available = get_available_memory()
    total = psutil.virtual_memory().total / 1024 / 1024 / 1024
    
    usage_pct = usage / total
    if usage_pct > MEMORY_THRESHOLD:
        logger.warning(f"Uso de memoria alto: {usage:.2f}GB ({usage_pct:.1%})")
        return True
    return False

def make_label(df: pd.DataFrame, h: int = 1) -> pd.Series:
    """Crear etiquetas para el horizonte especificado"""
    close_lead = df["close"].shift(-h)
    return (close_lead > df["close"]).astype(int)

def process_data_in_chunks(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE):
    """Procesar datos en chunks para manejar datasets grandes"""
    logger.info(f"Procesando dataset de {len(df)} filas en chunks de {chunk_size}")
    
    chunks = []
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
        
        # Liberar memoria periódicamente
        if i % (chunk_size * 5) == 0:
            gc.collect()
            if check_memory_limit():
                logger.warning("Límite de memoria alcanzado, forzando garbage collection")
                gc.collect()
    
    logger.info(f"Dataset dividido en {len(chunks)} chunks")
    return chunks

def create_walk_forward_splits(df: pd.DataFrame, n_splits: int = 5, embargo_minutes: int = 30):
    """Crear splits walk-forward con embargo temporal"""
    logger.info(f"Creando {n_splits} splits walk-forward con embargo de {embargo_minutes} minutos")
    
    # Convertir timestamps a datetime si es necesario
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ordenar por timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calcular embargo en barras (aproximado)
    time_diff = df_sorted['timestamp'].diff().median()
    embargo_bars = max(1, int(embargo_minutes / time_diff.total_seconds() * 60))
    
    logger.info(f"Embargo temporal: {embargo_bars} barras")
    
    # Crear splits
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo_bars)
    
    splits = []
    for train_idx, test_idx in tscv.split(df_sorted):
        train_data = df_sorted.iloc[train_idx]
        test_data = df_sorted.iloc[test_idx]
        
        splits.append({
            'train': train_data,
            'test': test_data,
            'train_start': train_data['timestamp'].min(),
            'train_end': train_data['timestamp'].max(),
            'test_start': test_data['timestamp'].min(),
            'test_end': test_data['timestamp'].max()
        })
    
    return splits

def train_fold(X_train, y_train, X_test, y_test, fold_idx, total_folds, 
               max_iter: int = 2000, use_gpu: bool = False):
    """Entrenar un fold individual con monitoreo detallado"""
    logger.info(f"Entrenando fold {fold_idx + 1}/{total_folds}")
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurar modelo según disponibilidad de GPU
    if use_gpu:
        try:
            from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
            clf = CuMLLogisticRegression(
                max_iter=max_iter,
                random_state=42,
                class_weight='balanced'
            )
            logger.info("Usando GPU para entrenamiento")
        except ImportError:
            logger.warning("CuML no disponible, usando CPU")
            clf = LogisticRegression(
                max_iter=max_iter,
                solver='lbfgs',
                random_state=42,
                class_weight='balanced'
            )
    else:
        clf = LogisticRegression(
            max_iter=max_iter,
            solver='lbfgs',
            random_state=42,
            class_weight='balanced'
        )
    
    # Entrenar con monitoreo de progreso
    start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # Predicciones
    proba = clf.predict_proba(X_test_scaled)[:, 1]
    pred = (proba >= 0.5).astype(int)
    
    # Métricas
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    acc = accuracy_score(y_test, pred)
    
    logger.info(f"Fold {fold_idx + 1} completado en {train_time:.2f}s - AUC: {auc:.4f}, Brier: {brier:.4f}, Acc: {acc:.4f}")
    
    return {
        'model': clf,
        'scaler': scaler,
        'metrics': {
            'auc': float(auc),
            'brier': float(brier),
            'acc': float(acc),
            'train_time': train_time,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    }

def detect_gpu():
    """Detectar si hay GPU disponible"""
    try:
        import cupy
        return True
    except ImportError:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

def calculate_adaptive_max_iter(dataset_size: int):
    """Calcular max_iter adaptativo según el tamaño del dataset"""
    # Fórmula adaptativa: más datos = más iteraciones
    base_iter = 1000
    size_factor = min(3.0, dataset_size / 10000)  # Factor máximo de 3x
    return int(base_iter * size_factor)

def save_checkpoint(checkpoint_data: dict, checkpoint_path: str):
    """Guardar checkpoint del entrenamiento"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"Checkpoint guardado en {checkpoint_path}")

def load_checkpoint(checkpoint_path: str):
    """Cargar checkpoint si existe"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento optimizado de modelo de dirección")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--tf", default=TF)
    parser.add_argument("--horizon", type=int, default=H)
    parser.add_argument("--from", dest="dt_from", default=None)
    parser.add_argument("--to", dest="dt_to", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-bars", type=int, default=None, help="Límite de filas; 0 o None = usar todo el histórico")
    parser.add_argument("--n-splits", type=int, default=5, help="Número de folds para validación walk-forward")
    parser.add_argument("--embargo-minutes", type=int, default=30, help="Embargo temporal en minutos")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Tamaño de chunk para procesamiento")
    parser.add_argument("--resume", action="store_true", help="Reanudar desde último checkpoint")
    parser.add_argument("--use-gpu", action="store_true", help="Usar GPU si está disponible")
    args, _ = parser.parse_known_args()

    # Configurar directorio de logs
    os.makedirs("logs", exist_ok=True)
    
    sym = args.symbol
    tf = args.tf
    h = int(args.horizon)
    
    logger.info(f"Iniciando entrenamiento optimizado para {sym}-{tf} (H={h})")
    logger.info(f"Memoria inicial: {get_memory_usage():.2f}GB")
    
    set_global_seeds(args.seed)
    
    # Construir dataset
    logger.info("Construyendo dataset...")
    df = build_dataset(sym, tf, use_snapshots=True)
    if df.empty:
        logger.error("Dataset vacío.")
        return

    # Aplicar límite de filas si se especifica
    max_bars = args.max_bars
    if max_bars is None:
        cfg = load_yaml("config/ml/training.yaml")
        max_bars = (
            cfg.get("training", {})
               .get("modes", {})
               .get("historical", {})
               .get("max_bars", None)
        )
    if max_bars and int(max_bars) > 0:
        df = df.tail(int(max_bars))
    
    logger.info(f"Dataset total: {len(df)} filas (max_bars={max_bars})")
    
    # Crear etiquetas
    logger.info("Creando etiquetas...")
    df["y"] = make_label(df, h)
    df = df.dropna()
    
    if len(df) < 1000:
        logger.error("Dataset muy pequeño después de limpieza")
        return
    
    # Seleccionar features
    feat_cols = FEATURES[:]
    X = df[feat_cols].astype(float)
    y = df["y"].astype(int)
    
    logger.info(f"Features seleccionadas: {len(feat_cols)}")
    logger.info(f"Distribución de clases: {y.value_counts().to_dict()}")
    
    # Detectar GPU
    use_gpu = args.use_gpu and detect_gpu()
    if use_gpu:
        logger.info("GPU detectada y habilitada")
    else:
        logger.info("Usando CPU para entrenamiento")
    
    # Calcular max_iter adaptativo
    max_iter = calculate_adaptive_max_iter(len(df))
    logger.info(f"Max_iter adaptativo: {max_iter}")
    
    # Crear splits walk-forward
    splits = create_walk_forward_splits(df, args.n_splits, args.embargo_minutes)
    
    # Configurar checkpoint
    checkpoint_path = f"logs/checkpoint_{sym}_{tf}_H{h}.pkl"
    
    # Cargar checkpoint si se solicita
    checkpoint_data = None
    if args.resume:
        checkpoint_data = load_checkpoint(checkpoint_path)
        if checkpoint_data:
            logger.info("Checkpoint cargado, reanudando entrenamiento")
        else:
            logger.info("No se encontró checkpoint, iniciando desde cero")
    
    # Entrenar con validación walk-forward
    fold_results = []
    best_auc = 0
    best_model = None
    best_scaler = None
    no_improvement_count = 0
    
    logger.info(f"Iniciando entrenamiento con {len(splits)} folds")
    
    for fold_idx, split in enumerate(splits):
        # Verificar si ya procesamos este fold
        if checkpoint_data and fold_idx < len(checkpoint_data.get('completed_folds', [])):
            logger.info(f"Fold {fold_idx + 1} ya completado, saltando...")
            continue
        
        # Procesar datos del fold
        train_data = split['train']
        test_data = split['test']
        
        X_train = train_data[feat_cols].astype(float)
        y_train = train_data['y'].astype(int)
        X_test = test_data[feat_cols].astype(float)
        y_test = test_data['y'].astype(int)
        
        # Entrenar fold
        fold_result = train_fold(
            X_train, y_train, X_test, y_test, 
            fold_idx, len(splits), max_iter, use_gpu
        )
        
        fold_results.append(fold_result)
        
        # Verificar mejora
        current_auc = fold_result['metrics']['auc']
        if current_auc > best_auc:
            best_auc = current_auc
            best_model = fold_result['model']
            best_scaler = fold_result['scaler']
            no_improvement_count = 0
            logger.info(f"Nueva mejor AUC: {best_auc:.4f}")
        else:
            no_improvement_count += 1
            logger.warning(f"Sin mejora en AUC por {no_improvement_count} folds consecutivos")
        
        # Early stopping
        if no_improvement_count >= 3:
            logger.warning("Early stopping activado - sin mejora en 3 folds consecutivos")
            break
        
        # Guardar checkpoint
        checkpoint_data = {
            'completed_folds': list(range(fold_idx + 1)),
            'fold_results': fold_results,
            'best_auc': best_auc,
            'best_model': best_model,
            'best_scaler': best_scaler
        }
        save_checkpoint(checkpoint_data, checkpoint_path)
        
        # Liberar memoria
        gc.collect()
        
        # Monitoreo de memoria
        memory_usage = get_memory_usage()
        logger.info(f"Uso de memoria: {memory_usage:.2f}GB")
        
        if check_memory_limit():
            logger.warning("Límite de memoria alcanzado, forzando garbage collection")
            gc.collect()
    
    # Calcular métricas finales
    if not fold_results:
        logger.error("No se completó ningún fold")
        return
    
    # Promediar métricas entre folds
    avg_metrics = {
        'auc': np.mean([r['metrics']['auc'] for r in fold_results]),
        'brier': np.mean([r['metrics']['brier'] for r in fold_results]),
        'acc': np.mean([r['metrics']['acc'] for r in fold_results]),
        'n_train': sum([r['metrics']['n_train'] for r in fold_results]),
        'n_test': sum([r['metrics']['n_test'] for r in fold_results]),
        'n_folds': len(fold_results),
        'best_auc': best_auc
    }
    
    logger.info("Métricas finales:")
    for metric, value in avg_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Guardar mejor modelo
    if best_model is not None:
        os.makedirs("artifacts/direction", exist_ok=True)
        artifact_uri = f"artifacts/direction/{sym}_{tf}_H{h}_logreg_optimized.pkl"
        
        model_data = {
            'model': best_model,
            'scaler': best_scaler,
            'feature_names': feat_cols,
            'metrics': avg_metrics,
            'fold_results': fold_results,
            'training_config': {
                'n_splits': args.n_splits,
                'embargo_minutes': args.embargo_minutes,
                'max_iter': max_iter,
                'use_gpu': use_gpu,
                'chunk_size': args.chunk_size
            }
        }
        
        save_pickle(model_data, artifact_uri)
        logger.info(f"Modelo guardado en {artifact_uri}")
        
        # Registrar versión
        agent_id = get_or_create_agent("DirectionLogRegOptimized", "direction")
        ver_id = register_version(
            agent_id=agent_id,
            version="v2.0.0",
            params={
                "symbol": sym, 
                "timeframe": tf, 
                "horizon": h, 
                "model": "LogRegOptimized",
                "n_splits": args.n_splits,
                "embargo_minutes": args.embargo_minutes,
                "max_iter": max_iter,
                "use_gpu": use_gpu
            },
            artifact_uri=artifact_uri,
            train_start=df["timestamp"].min(),
            train_end=df["timestamp"].max(),
            metrics=avg_metrics,
            promoted=False
        )
        logger.info(f"Versión registrada con ID: {ver_id}")
        
        # Limpiar checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("Checkpoint limpiado")
    
    # Limpieza final
    gc.collect()
    final_memory = get_memory_usage()
    logger.info(f"Entrenamiento completado. Memoria final: {final_memory:.2f}GB")
    
    print("metrics:", avg_metrics)
    print("registered version id:", ver_id if 'ver_id' in locals() else "N/A")

if __name__ == "__main__":
    main()