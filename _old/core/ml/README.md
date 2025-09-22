core\ml\README.md

# core/ml/ — Arquitectura de Agentes, Entrenamiento y Predicción

Este módulo define cómo entrenamos, versionamos y servimos agentes de IA/ML para el bot. Está diseñado para trabajar sobre la base de datos PostgreSQL que ya usamos (OHLCV, features, señales, órdenes) y para escalar multi-símbolo y multi-timeframe.

## 🎯 Objetivos

- **Agentes modulares** (dirección, régimen, volatilidad, ejecución, ensemble) con instancias por símbolo/TF
- **Entrenamiento batch reproducible** con champion/challenger y promoción a producción
- **Inferencia en tiempo real** sincronizada con el cierre de barras (1m y 5m), leyendo/escribiendo de la DB
- **Trazabilidad total**: cada predicción, señal y trade queda ligado a la versión del agente que la generó

## 📁 Estructura

```
core/ml/
├─ agents/                     # Implementaciones de agentes (plantillas)
│  ├─ base_agent.py            # Interfaz BaseAgent (contrato)
│  ├─ direction_xgb.py         # Agente direccional (XGBoost/LightGBM)
│  ├─ regime_kmeans.py         # Agente de régimen (KMeans/HMM)
│  ├─ volatility_quantile.py   # Agente de volatilidad (quantile/GARCH)
│  └─ execution_rules.py       # Agente de ejecución (reglas; luego RL/bandits)
│
├─ ensemble/
│  └─ weighted.py              # Combinación de señales (side/strength)
│
├─ datasets/
│  ├─ builder.py               # JOIN OHLCV+Features, snapshots multi-TF
│  └─ schema.md                # Especificación de columnas/etiquetas
│
├─ training/
│  ├─ train_direction.py       # Entrena direccional (walk-forward)
│  ├─ train_regime.py          # Entrena régimen
│  ├─ train_volatility.py      # Entrena volatilidad (SL/TP)
│  ├─ registry.py              # Alta de modelos/agentes en DB (versionado)
│  └─ model_card_template.md   # Plantilla de reporte por versión
│
├─ inference/
│  ├─ infer_realtime.py        # Loop de inferencia (lee DB, escribe predicciones)
│  └─ postprocess.py           # Umbrales, calibración por símbolo, etc.
│
├─ backtests/
│  ├─ run_backtest.py          # Backtest con AgentSignals → PnL/Sharpe/DD
│  └─ analyzers.py             # Métricas, gráficos
│
├─ utils/
│  ├─ time.py                  # utilidades TF, ventanas
│  ├─ io.py                    # carga/guardado de artefactos (.pkl/.json)
│  └─ seeds.py                 # seeds y control de aleatoriedad
│
└─ README.md                   # este documento
```

Los artefactos de modelos (.pkl) se guardan fuera del código, p.ej. en `artifacts/`, y su ruta queda registrada en la DB.

## 🧩 Tipos de agentes y contrato

### BaseAgent (contrato)

Cada agente implementa:
- `name`, `kind` (direction|regime|volatility|execution|ensemble), `version`
- `load(artifact_uri)` → carga pesos/estado desde disco
- `predict(row: dict) -> dict` → devuelve un payload (JSON serializable)

### Payload esperado por tipo:

- **direction**: `{"prob_up": float}` (0..1) a un horizonte H (p.ej., 1 barra)
- **regime**: `{"regime": "trend|mean_revert|chop", "gate": bool}`
- **volatility**: `{"sigma": float, "tp": float, "sl": float}` (en unidades de precio o ATR-multipliers)
- **execution**: `{"side": -1|0|1, "size": float, "sl": float, "tp": float, "meta": {...}}`
- **ensemble**: combina entradas anteriores y devuelve al menos `{"side": -1|0|1, "strength": float}`

**Instancias**: una plantilla (p.ej. DirectionXGB v1) puede tener múltiples instancias por símbolo/TF (BTC-1m, ETH-5m, …).

## 🗃️ Contratos de datos (DB)

Todo queda en PostgreSQL; la inferencia y entrenamiento leen/escriben ahí.

### Tablas de entrada:
- `trading.HistoricalData` y `trading.Features`: entrada

### Tablas del módulo ML:
- `trading.Agents`: catálogo de agentes (direccional, régimen, etc.)
- `trading.AgentVersions`: versiones entrenadas (parámetros, métricas, artifact_uri)
- `trading.AgentPreds`: predicciones crudas por barra `{prob_up, regime, sigma, …}`
- `trading.AgentSignals`: señal combinada por barra (side/strength/sl/tp) → la consume el OMS
- *(Opcional)* `trading.Models/trading.Predictions`: si separas "modelo" de "agente"

### Claves y unicidad recomendadas:
- **AgentPreds**: `UNIQUE(agent_version_id, symbol, timeframe, timestamp, horizon)`
- **AgentSignals**: `UNIQUE(symbol, timeframe, timestamp)`

## 🧠 Flujo lógico (de 0 a operación)

### 1. Dataset (por símbolo/TF de ejecución):
- JOIN `HistoricalData` + `Features` por `symbol,timeframe,timestamp`
- **Snapshots multi-TF**: añadir a cada fila 1m/5m las últimas lecturas cerradas de 15m/1h/4h/1d (EMA slopes, RSI, ST, SMC si aplicase)
- **Label (dirección)**: p.ej., `y = 1[close_{t+1} > close_t]` (o retorno a H barras)

### 2. Entrenamiento (batch, reproducible):
- Split temporal (walk-forward); métricas: AUC, Brier, accuracy y PnL/Sharpe/DD en backtest simple
- Guardar artefacto (.pkl), metrics JSON, ventana temporal (train/test), commit del repo
- Registrar en `AgentVersions` (challenger)
- **Promoción a campeón** si supera umbrales (métricas y riesgo)

### 3. Inferencia en tiempo real (cada cierre de 1m/5m):
- **Direccional** produce `prob_up`
- **Régimen** produce `gate`
- **Volatilidad** produce `sigma/sl/tp`
- **Ensemble** combina → `AgentSignals` (side/strength/sl/tp)
- Gestor de riesgo global (fuera de ML) aplica límites (exposición total, por símbolo, correlación)
- OMS coloca órdenes → fills → `trading.Trades`

### 4. Monitoreo y re-entrenos:
- Jobs nocturnos que re-entrenan (ventana móvil), validan y, si procede, promueven nuevos agentes

## ⚙️ Entrenamiento: convenciones

### Reproducibilidad y configuración:
- **Semillas**: fija seeds en `utils/seeds.py` (NumPy/ML libs) para reproducibilidad
- **Ventanas**: define horizontes por TF (1m/5m) y lookbacks razonables (90–180 días)
- **Calibración por símbolo**: umbrales de probabilidad (ej. entrar long si `prob_up ≥ 0.55`), ajustados por activo
- **Regularización**: prioriza modelos tabulares (XGBoost/LightGBM) antes de deep nets para arranque estable

### Regímenes y gates:
- **Regímenes**: k-means/HMM en TF altos (4h/1d); produce `gate` para apagar en "chop"

### Champion/Challenger:
- El **challenger** se evalúa exactamente sobre el último período hold-out
- **Promociona** si mejora PnL y no empeora MaxDD/Calmar por debajo de tolerancias
- Todo queda registrado (quién, cuándo, datos usados, métricas, artefacto)

## 🚀 Inferencia: ciclo en vivo

### Trigger y sincronización:
1. **Dispara en el cierre** de la barra más corta (1m): ya tienes OHLCV + Features (updaters)
2. **Construye snapshot multi-TF** (vista/materializada o en memoria)
3. **Ejecuta agentes por símbolo**:
   - Direccional (1m/5m), Régimen (4h/1d), Volatilidad (símbolo)
4. **Aplica ensemble** y escribe `AgentSignals`
5. **El OMS lee** `AgentSignals` recientes y envía órdenes

### Paralelismo:
Es natural correr todas las instancias (BTC, ETH, SOL, …) en paralelo; compartiendo el mismo artefacto si el modelo es "global 1m" con calibración por símbolo.

## 🧪 Backtesting y métricas

### Pipeline de evaluación:
- **Entrada**: `AgentSignals` → simulador (slippage, fees, latency)
- **Salida**: PnL, Sharpe/Sortino/Calmar, MaxDD, PF, win-rate
- **Persistencia**: métricas de cada run ligadas a la versión del conjunto (y a cada agente si quieres granularidad)
- **Comparaciones**: curva del campeón vs challenger; ranking de estrategias (TOP-N / BOTTOM-N)

## 🔐 Versionado y trazabilidad

### Artefactos y registros:
- **Artefactos**: `artifacts/<agent>/<symbol|global>/<tf>/<version>.pkl` (+ hash)
- **DB**: `AgentVersions` guarda `artifact_uri`, fechas de train, métricas, params
- **Predicciones/Señales**: cada fila referencia `agent_version_id`
- **Model cards**: un `.md` por versión con: dataset, split, hiperparámetros, métricas, riesgos conocidos, "dónde usar"

## 🧱 Dependencias recomendadas

```python
# ML Core
xgboost                # o lightgbm (direccional / cuantiles)
scikit-learn          # split, métricas, k-means
hmmlearn              # si HMM para régimen
statsmodels           # si GARCH

# Data & DB
pandas
numpy
sqlalchemy
psycopg2
```

## 📝 Convenciones de naming

### Estándar de nomenclatura:
- **Agentes**: `DirectionXGB`, `RegimeKMeans`, `VolQuantile`, `ExecRules`, `EnsembleWeighted`
- **Versiones**: `vMAJOR.MINOR.PATCH` (ej. `v1.3.0`)
- **Instancias**: `<AgentName>@<SYMBOL>-<TF>` (ej. `DirectionXGB@BTCUSDT-1m`)

## ✅ Checklist para implementar

### Fase 1: Infraestructura
- [ ] Crear tablas: `Agents`, `AgentVersions`, `AgentPreds`, `AgentSignals`
- [ ] Escribir `BaseAgent` y stubs (clases vacías con firma `load/predict`)
- [ ] `datasets/builder.py`: ensamblar dataset y snapshots multi-TF

### Fase 2: Entrenamiento
- [ ] `training/train_direction.py`: entrenar direccional (walk-forward) + registrar versión
- [ ] `backtests/run_backtest.py`: medir PnL y comparar champion vs challenger
- [ ] Definir reglas de promoción (umbrales) y un job batch nocturno

### Fase 3: Inferencia
- [ ] `inference/infer_realtime.py`: loop de inferencia en cierre de barra (graba `AgentPreds/AgentSignals`)

## ❗ Notas operativas

### Persistencia y escalabilidad:
- **Todo contra la DB**: evita archivos sueltos excepto artefactos de modelos
- **Retención/particionado**: `AgentPreds` puede crecer mucho; preparar particiones por fecha si hace falta

### Seguridad y auditoría:
- **Seguridad**: los agentes no guardan claves API; solo leen/escriben en DB y entregan señales al OMS
- **Auditoría**: cualquier cambio de versión queda en `AgentVersions` con métricas y `artifact_uri`

## 🔗 Integración con el sistema

### Con el data layer:
- Lee de `trading.HistoricalData` y `trading.Features`
- Produce `trading.AgentSignals` para el OMS

### Con el trading engine:
- Las señales se consumen en `core/trading/decision_maker.py`
- El risk manager aplica límites globales antes de ejecutar

### Con el control system:
- Comandos Telegram pueden triggear re-entrenamientos
- Métricas de performance se reportan via `/status`