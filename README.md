# Bot Trading v11 - Scalping Autónomo de Futuros con IA

## 🎮 Comandos de Control Rápido

### 📊 Gestión de Datos
```bash
# Descargar datos históricos (una sola vez)
python core/data/historical_downloader.py

# Actualizar datos en tiempo real (continuo)
python core/data/realtime_updater.py

# Una pasada de actualización de datos
python core/data/realtime_updater.py --once
```

### 🧮 Gestión de Features/Indicadores
```bash
# Calcular indicadores iniciales (una sola vez)
python core/features/indicator_calculator.py

# Actualizar indicadores continuamente (recomendado)
python core/features/features_updater.py

# Una pasada de actualización de indicadores
python core/features/features_updater.py --once
```

### 🤖 Control del Bot
```bash
# Iniciar bot completo
python main.py

# Ver estado del sistema
python scripts/reporting/status_report.py

# Entrenar agentes ML
python scripts/ml/train_agents.py

# Reentrenar agentes
python scripts/ml/retrain_agents.py
```

### 🗄️ Base de Datos
```bash
# Inicializar base de datos
python scripts/initialization/init_db.py

# Verificar conexión a DB
python -c "from core.data.database import ENGINE; print('DB OK' if ENGINE else 'DB Error')"
```

## 🚀 Visión General

Bot Trading v11 es un sistema autónomo para trading de futuros perpetuos en criptomonedas, enfocado en scalping (1m/5m) con confirmaciones de tendencias en timeframes superiores (15m, 1h, 4h, 1d). Utiliza aprendizaje por refuerzo (PPO) con agentes por símbolo que operan en paralelo, optimizando el balance según volatilidad y riesgo para maximizar rentabilidad. Controlado via Telegram, almacena datos en PostgreSQL con backups en .csv, y guarda las mejores/peores estrategias para aprendizaje continuo. Diseñado para ser robusto, escalable y "enterprise-grade".

### Características Principales
- 🎯 **Futuros Perpetuos**: Opera con leverage dinámico (e.g., min: 5, max: 80) configurado por símbolo
- ⚡ **Scalping Autónomo**: Trades en 1m/5m, confirmados por tendencias multi-TF
- 🧠 **IA/ML**: Agentes PPO por símbolo, entrenados en paralelo, usan datos históricos y en tiempo real, almacenan las mejores/peores 1000 estrategias
- 💰 **Gestión de Balance**: Divide capital según volatilidad (ATR) y riesgo (máx 1% por trade)
- 📊 **Datos**: PostgreSQL para OHLCV de todos los TFs (1m, 5m, 15m, 1h, 4h, 1d); backups en `data/{symbol}/{symbol}_{tf}.csv`
- 📱 **Control Telegram**: Comandos para iniciar, detener, reentrenar, ver métricas; alertas en tiempo real
- 🎯 **Estrategias**: Cada agente guarda estrategias (TFs, indicadores, herramientas, PnL) en DB para auditoría y aprendizaje

**Símbolos Iniciales**: BTCUSDT, ETHUSDT, ADAUSDT, SOLUSDT, DOGEUSDT, LINKUSDT (futuros perpetuos en BitGet)

## 📁 Estructura del Proyecto

```
bot_trading_v11/
├── agents/                     # Modelos ML entrenados (vacío - se crean al entrenar)
├── automaticos/               # Scripts de automatización
│   └── start_updater.ps1     # Script PowerShell para iniciar actualizador
├── config/                     # Configuraciones
│   ├── .env                   # Variables de entorno (no en git)
│   ├── trading/               # Configuraciones de trading
│   │   ├── symbols.yaml       # Símbolos, TFs, leverage (min/max)
│   │   └── risk.yaml          # Gestión de riesgos
│   ├── ml/                    # Configuraciones de ML
│   │   ├── training.yaml      # Modos y objetivos de entrenamiento
│   │   └── rewards.yaml       # Recompensas/penalizaciones RL
│   └── system/                # Configuraciones del sistema
│       ├── paths.yaml         # Rutas para datos, agentes
│       ├── logging.yaml       # Configuración de logging
│       ├── monitoring.yaml    # Configuración de monitoreo
│       └── telegram.yaml      # Configuración de Telegram
├── core/                      # Lógica principal
│   ├── data/                  # Manejo de datos
│   │   ├── database.py        # Esquema PostgreSQL y ORM
│   │   ├── historical_downloader.py  # Descarga histórica (BitGet)
│   │   ├── realtime_updater.py  # Actualización en tiempo real
│   │   └── README.md          # Documentación del módulo de datos
│   ├── features/              # Cálculo de indicadores técnicos
│   │   ├── indicator_calculator.py  # Calculador de indicadores
│   │   ├── features_updater.py     # Actualizador continuo de features
│   │   └── README.md          # Documentación del módulo de features
│   ├── ml/                    # Machine Learning (vacío - pendiente implementación)
│   ├── trading/               # Lógica de trading (vacío - pendiente implementación)
│   └── control/               # Interfaz de control (vacío - pendiente implementación)
├── data/                      # Backups y datos procesados (vacío - se crea al ejecutar)
├── db/                        # Migraciones de DB
│   └── migrations/
│       └── versions/          # Versiones de migraciones (vacío)
├── scripts/                   # Scripts para comandos y automatización
│   ├── initialization/        # Tareas iniciales
│   │   ├── init_db.sql        # Script SQL de inicialización
│   │   └── README.md          # Documentación de inicialización
│   ├── ml/                    # Scripts de ML (vacío - pendiente implementación)
│   ├── ps/                    # Scripts de PowerShell (vacío)
│   ├── reporting/             # Scripts de reportes (vacío - pendiente implementación)
│   └── trading/               # Scripts de trading (vacío - pendiente implementación)
├── tests/                     # Tests unitarios (vacío - pendiente implementación)
├── docs/                      # Documentación (vacío)
├── venv/                      # Entorno virtual Python
├── .gitignore                 # Archivos ignorados por Git
├── requirements.txt           # Dependencias Python
├── setup_environment.md       # Guía de configuración del entorno
├── test_setup.py             # Script de prueba de configuración
└── README.md                  # Este archivo
```

### 📊 Estado Actual de Implementación

#### ✅ **Completamente Implementado**
- **Data Layer**: Descarga histórica, actualización en tiempo real, base de datos
- **Features Module**: Cálculo de indicadores técnicos, actualización continua
- **Configuración**: Archivos YAML y variables de entorno
- **Documentación**: READMEs detallados para cada módulo

#### 🚧 **Pendiente de Implementación**
- **ML Module**: Agentes PPO, entrenamiento, autolearn
- **Trading Module**: Motor de trading, gestión de riesgos, decisiones
- **Control Module**: Interfaz de Telegram, comandos
- **Scripts**: Comandos de trading, reportes, ML
- **Tests**: Tests unitarios y de integración

#### 📁 **Directorios Vacíos (Se Crean Dinámicamente)**
- `agents/` - Modelos ML entrenados
- `data/` - Backups CSV de datos
- `tests/` - Tests unitarios
- `docs/` - Documentación adicional

## 🔄 Protocolos del Sistema

### 1. Descarga y Almacenamiento de Datos

- **Históricos**: `scripts/initialization/download_historical.py` usa `core/data/historical_downloader.py` para descargar OHLCV de futuros perpetuos (BitGet, 1+ año) para todos los TFs (1m, 5m, 15m, 1h, 4h, 1d). Guarda en PostgreSQL (`HistoricalData`) y respalda en `data/{symbol}/{symbol}_{tf}.csv`
- **Tiempo Real**: `core/data/realtime_fetcher.py` recolecta OHLCV de todos los TFs via WebSocket de BitGet, guarda en PostgreSQL y actualiza .csv via `core/data/data_updater.py`
- **Alineación**: `core/data/timeframe_aligner.py` sincroniza TFs por timestamp, generando `data/{symbol}/{symbol}_aligned.csv` para ML
- **Actualización**: `core/data/data_updater.py` añade nuevos datos sin duplicados, verificando timestamps
- **Integridad**: Índices en PostgreSQL (symbol, timeframe, timestamp) aseguran queries rápidas

### 2. Entrenamiento de Agentes

- **Paralelismo**: Agentes PPO por símbolo entrenan en paralelo (`core/ml/reinforcement_agent.py`) usando multiprocessing
- **Entorno Gym**: Simula trading de futuros (estados: features multi-TF; acciones: long, short, hold; recompensas: PnL)
- **Histórico como Referencia**: `core/ml/reinforcement_agent.py` y `core/ml/autolearn.py` usan datos históricos para entrenar y validar patrones (precios, indicadores)
- **Entrenamiento Inicial**: `scripts/ml/train_agents.py` entrena con histórico según `config/ml/training.yaml`
- **Entrenamiento en Tiempo Real**: `scripts/ml/retrain_agents.py` combina histórico y datos en tiempo real, ajusta modelos semanalmente o si Sharpe <1.5
- **Estrategias**: Cada agente guarda las mejores/peores 1000 estrategias en `MLStrategies` (TFs, indicadores, herramientas, PnL)

### 3. Trading de Futuros

- **Ejecución**: `core/trading/trading_engine.py` maneja longs, shorts, fees, slippage, y leverage dinámico (de `config/trading/symbols.yaml`, e.g., min: 5, max: 80)
- **Decisiones**: `core/trading/decision_maker.py` usa señales ML y tendencias históricas (EMA, RSI) de 15m/1h/4h/1d para trades en 1m/5m
- **Riesgo**: `core/trading/risk_manager.py` limita posición a 1% capital, usa SL/TP (ATR), detiene trading si drawdown >5%
- **Balance**: Agentes dividen capital según ATR y riesgo, optimizando rentabilidad en paralelo
- **Histórico**: `core/trading/decision_maker.py` consulta datos históricos para predecir precios y validar indicadores

### 4. Almacenamiento de Estrategias

#### Tabla MLStrategies
- **Columnas**: 
  - `id` (SERIAL, PK)
  - `symbol` (VARCHAR)
  - `timestamp` (TIMESTAMP)
  - `action` (VARCHAR)
  - `timeframes` (JSON)
  - `indicators` (JSON)
  - `tools` (JSON)
  - `leverage` (NUMERIC)
  - `pnl` (NUMERIC)
  - `performance` (NUMERIC)
  - `outcome` (VARCHAR, best/worst)
- **Índices**: symbol, timestamp

#### Uso y Consulta
- **Uso**: `core/ml/autolearn.py` analiza estrategias para evitar errores y replicar aciertos
- **Consulta**: `scripts/reporting/fetch_logs.py` permite ver estrategias via Telegram (`/logs`)

### 5. Control Telegram

- **Comandos**: `/status`, `/start_trading`, `/stop`, `/retrain`, `/logs`, `/init_db`, `/download_historical`
- **Orquestación**: `core/control/telegram_interface.py` ejecuta scripts en `scripts/` (e.g., `reporting/status_report.py`)
- **Métricas**: Envía balance, PnL, drawdown, posiciones, gráficos, y estrategias
- **Alertas**: Notifica trades, errores, drawdowns

### 6. Despliegue

- **VPS**: Docker (`Dockerfile`) para AWS EC2, DigitalOcean
- **Logging**: Configurado en `config/system/logging.yaml` (rotación diaria, consola + archivo)

## 🗄️ Base de Datos (PostgreSQL)

**Motor**: PostgreSQL 13+

### Tablas (en `core/data/database.py`)

#### HistoricalData
- **Columnas**: `id` (SERIAL, PK), `symbol` (VARCHAR), `timeframe` (VARCHAR), `timestamp` (TIMESTAMP), `open` (NUMERIC), `high` (NUMERIC), `low` (NUMERIC), `close` (NUMERIC), `volume` (NUMERIC)

#### Trades
- **Columnas**: `id` (SERIAL, PK), `symbol` (VARCHAR), `side` (VARCHAR), `quantity` (NUMERIC), `price` (NUMERIC), `pnl` (NUMERIC), `timestamp` (TIMESTAMP)

#### MLStrategies
- **Columnas**: `id` (SERIAL, PK), `symbol` (VARCHAR), `timestamp` (TIMESTAMP), `action` (VARCHAR), `timeframes` (JSON), `indicators` (JSON), `tools` (JSON), `leverage` (NUMERIC), `pnl` (NUMERIC), `performance` (NUMERIC), `outcome` (VARCHAR)

### Gestión
- **Migraciones**: Gestionadas con Alembic (`db/migrations/`)
- **Backup**: `data/{symbol}/{symbol}_{tf}.csv` para redundancia

## 🛠️ Tecnologías y Herramientas

### Lenguaje Principal
**Python 3.10+**

### Librerías Clave
- **CCXT**: BitGet (futuros perpetuos)
- **SQLAlchemy**: ORM para PostgreSQL
- **Stable Baselines3**: PPO para RL
- **TA-Lib**: Indicadores (RSI, MACD, Bollinger, EMA)
- **python-telegram-bot**: Control Telegram
- **structlog**: Logging estructurado
- **PyYAML**: Manejo de YAML
- **multiprocessing**: Paralelismo de agentes

### Infraestructura
- **Docker**: Containerización
- **PostgreSQL**: Base de datos
- **VPS**: AWS EC2, DigitalOcean

### Testing
**Pytest** (`tests/`)

## ⚙️ Instalación

### 1. Clonar Repositorio
```bash
git clone https://github.com/tuusuario/bot_trading_v11.git
cd bot_trading_v11
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Entorno

Copiar `config/.env.example` a `config/.env`, editar:

```env
DB_URL=postgresql://user:pass@localhost/trading_db
BITGET_API_KEY=tu_api_key
BITGET_SECRET=tu_secret
TELEGRAM_TOKEN=tu_token
ALLOWED_USER_ID=tu_telegram_id
```

Editar YAMLs en `config/trading/`, `config/ml/`, `config/system/`

### 5. Inicializar Sistema
```bash
# Inicializar DB
python scripts/initialization/init_db.py

# Descargar datos históricos
python scripts/initialization/download_historical.py

# Entrenar agentes
python scripts/ml/train_agents.py

# Lanzar bot
python main.py
```

## 📱 Uso via Telegram

### Comandos Disponibles
- `/start`: Bienvenida
- `/status`: Balance, PnL, drawdown, posiciones
- `/start_trading`: Inicia trading
- `/stop`: Detiene trading
- `/retrain`: Reentrena agentes
- `/logs`: Trades, estrategias, logs
- `/init_db`: Inicializa DB
- `/download_historical`: Descarga datos

### Seguridad
- **Autenticación**: Solo `ALLOWED_USER_ID` en `.env`
- **Alertas**: Trades, errores, drawdowns

## 💰 Gestión de Balance y Rentabilidad

### Estrategia de Capital
- **Paralelismo**: Agentes por símbolo operan en paralelo (multiprocessing)
- **Balance**: Divide capital según ATR y riesgo (máx 1% por trade)
- **Leverage**: Dinámico dentro de min/max (`config/trading/symbols.yaml`)
- **Rentabilidad**: Optimiza trades en 1m/5m con confirmaciones multi-TF, usando histórico para predicción

### Sistema de Estrategias
- **Almacenamiento**: Mejores/peores 1000 estrategias por agente en `MLStrategies`
- **Aprendizaje**: Análisis continuo para evitar errores y replicar aciertos
- **Auditoría**: Registro completo de TFs, indicadores, herramientas y PnL utilizados

## 🧪 Desarrollo

### Testing
```bash
pytest tests/
```

### Logging
Configurado en `config/system/logging.yaml`

### Contribuciones
Issues/PRs en el repositorio

## 🚢 Despliegue

### Docker

#### Construir imagen
```bash
docker build -t bot_trading_v11 .
```

#### Ejecutar
```bash
docker run -d --env-file config/.env bot_trading_v11
```

### Monitoreo
Control completo via Telegram

## 📝 Notas Importantes

- **Futuros**: Configura leverage en `config/trading/symbols.yaml`
- **Estrategias**: Mejores/peores 1000 estrategias por agente en `MLStrategies`
- **Escalabilidad**: Añade símbolos/TFs en `config/trading/symbols.yaml`
- **Aprendizaje Continuo**: Sistema de estrategias para optimización constante
- **Enterprise-Grade**: Robusto, escalable y con gestión profesional de riesgos

---

## 🎯 ¡Automatiza tu trading de futuros con IA robusta y rentable!

**Sistema avanzado con aprendizaje continuo y almacenamiento inteligente de estrategias para máxima rentabilidad.**