import os, pickle, pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

from core.ml.datasets.builder import build_dataset, FEATURES
from core.ml.training.registry import get_or_create_agent, register_version
from core.ml.utils.io import save_pickle
from core.ml.utils.seeds import set_global_seeds

load_dotenv("config/.env")
ENGINE = create_engine(os.getenv("DB_URL"))

SYMBOL = os.getenv("ML_SYMBOL", "BTCUSDT")
TF     = os.getenv("ML_TF", "1m")
H      = int(os.getenv("ML_H", "1"))  # horizonte en barras

def make_label(df: pd.DataFrame, h: int = 1) -> pd.Series:
    close_lead = df["close"].shift(-h)
    return (close_lead > df["close"]).astype(int)

def main():
    set_global_seeds(42)
    df = build_dataset(SYMBOL, TF, use_snapshots=True)
    if df.empty:
        print("Dataset vacío.")
        return

    # label y features (solo filas completas)
    df["y"] = make_label(df, H)
    df = df.dropna()
    feat_cols = FEATURES[:]  # solo TF base + (si añadiste snapshots, puedes incluir los sufijos)
    # Para empezar simple: sin sufijos de TF alto. Luego los añadimos.
    X = df[feat_cols].astype(float)
    y = df["y"].astype(int)

    # Split temporal simple (80/20)
    n = len(df)
    cut = int(n*0.8)
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr, yte = y.iloc[:cut], y.iloc[cut:]

    # Escalar datos para mejor convergencia
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    # Modelo con parámetros optimizados
    clf = LogisticRegression(
        max_iter=2000,  # Aumentar iteraciones para convergencia
        solver='lbfgs',  # Explícito para claridad
        random_state=42,  # Reproducibilidad
        class_weight='balanced'  # Balancear clases si están desbalanceadas
    )
    clf.fit(Xtr_scaled, ytr)
    proba = clf.predict_proba(Xte_scaled)[:,1]
    pred  = (proba >= 0.5).astype(int)  # Umbral más estándar

    metrics = {
        "auc": float(roc_auc_score(yte, proba)),
        "brier": float(brier_score_loss(yte, proba)),
        "acc": float((pred == yte).mean()),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte))
    }
    print("metrics:", metrics)

    # Guardar artefacto + registrar versión
    os.makedirs("artifacts/direction", exist_ok=True)
    artifact_uri = f"artifacts/direction/{SYMBOL}_{TF}_H{H}_logreg.pkl"
    
    # Guardar modelo y scaler juntos
    model_data = {
        'model': clf,
        'scaler': scaler,
        'feature_names': feat_cols
    }
    save_pickle(model_data, artifact_uri)

    agent_id = get_or_create_agent("DirectionLogReg", "direction")
    ver_id   = register_version(
        agent_id=agent_id,
        version="v1.0.0",
        params={"symbol": SYMBOL, "tf": TF, "h": H, "model":"LogReg"},
        artifact_uri=artifact_uri,
        train_start=df["timestamp"].min(),
        train_end=df["timestamp"].max(),
        metrics=metrics,
        promoted=False
    )
    print("registered version id:", ver_id)

if __name__ == "__main__":
    main()
