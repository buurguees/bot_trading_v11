core\features\README.md

# Features Module - Cálculo y Actualización de Indicadores Técnicos

Este módulo se encarga del cálculo y actualización continua de indicadores técnicos para todos los símbolos y timeframes configurados. Proporciona una capa de procesamiento robusta que convierte datos OHLCV en features utilizables por los agentes de ML.

## 🧭 Visión General

```
HistoricalData (OHLCV) ──▶ indicator_calculator.py ──▶ Features (indicadores)
                                │
                                ▼
                        features_updater.py ──▶ Actualización continua
```

## 📁 Archivos del Módulo

### `indicator_calculator.py`
**Calculador principal de indicadores técnicos**

**Funciones principales:**
- `fetch_candles()` - Obtiene datos OHLCV desde PostgreSQL
- `compute_and_save()` - Calcula indicadores y los guarda en la tabla Features
- `read_symbols_and_tfs()` - Lee configuración de símbolos y timeframes

**Indicadores implementados:**
- **RSI** (14 períodos) - Momentum oscillator
- **EMA** (20, 50, 200) - Medias móviles exponenciales
- **MACD** (12, 26, 9) - Convergencia/divergencia de medias móviles
- **ATR** (14 períodos) - Average True Range para volatilidad
- **Bollinger Bands** (20, 2.0) - Bandas de volatilidad
- **OBV** - On Balance Volume
- **Supertrend** (10, 3.0) - Indicador de tendencia

**Características de seguridad:**
- Conversión explícita a `float` para evitar problemas con tipos `Decimal`
- Doble capa de validación de tipos de datos
- Manejo robusto de errores y datos faltantes

### `features_updater.py`
**Actualizador continuo de features**

**Funciones principales:**
- `run_loop()` - Bucle principal de actualización continua
- `one_pass_all()` - Ejecuta una pasada completa por todos los símbolos/TFs
- `next_bar_sleep_seconds()` - Calcula tiempo de espera hasta el próximo cierre de vela

**Características:**
- Actualización incremental (solo nuevas barras)
- Sincronización con cierres de velas
- Jitter aleatorio para distribuir carga
- Grace period para evitar actualizaciones prematuras

## 🗄️ Esquema de la Tabla Features

**Tabla:** `trading.Features`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| id | BIGSERIAL (PK) | Identificador único |
| symbol | VARCHAR | Símbolo del activo |
| timeframe | VARCHAR(5) | Timeframe (1m, 5m, 15m, 1h, 4h, 1d) |
| timestamp | TIMESTAMPTZ | Timestamp de la vela |
| rsi14 | NUMERIC | RSI de 14 períodos |
| ema20 | NUMERIC | EMA de 20 períodos |
| ema50 | NUMERIC | EMA de 50 períodos |
| ema200 | NUMERIC | EMA de 200 períodos |
| macd | NUMERIC | Línea MACD |
| macd_signal | NUMERIC | Señal MACD |
| macd_hist | NUMERIC | Histograma MACD |
| atr14 | NUMERIC | ATR de 14 períodos |
| bb_mid | NUMERIC | Banda media de Bollinger |
| bb_upper | NUMERIC | Banda superior de Bollinger |
| bb_lower | NUMERIC | Banda inferior de Bollinger |
| obv | NUMERIC | On Balance Volume |
| supertrend | NUMERIC | Valor Supertrend |
| st_dir | INTEGER | Dirección Supertrend (1/-1) |
| created_at | TIMESTAMPTZ | Fecha de creación |
| updated_at | TIMESTAMPTZ | Fecha de actualización |

**Constraint:** `UNIQUE(symbol, timeframe, timestamp)`  
**Índice:** `idx_features_symbol_tf_ts(symbol, timeframe, timestamp DESC)`

## 🚀 Comandos de Ejecución

### Cálculo Inicial de Features
```bash
# Calcular features para todos los símbolos/TFs (una sola pasada)
python core/features/indicator_calculator.py
```

### Actualización Continua de Features
```bash
# Actualización continua (recomendado para producción)
python core/features/features_updater.py

# Una sola pasada de actualización
python core/features/features_updater.py --once
```

### Variables de Entorno
```bash
# Configuración de logging
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR

# Configuración de timing (milisegundos)
POLL_GRACE_MS=5000      # Tiempo de gracia tras cierre de vela
JITTER_MAX_MS=1500      # Jitter aleatorio para distribuir carga
```

## ⚙️ Configuración

### Archivo de Configuración
El módulo lee la configuración desde `config/trading/symbols.yaml`:

```yaml
defaults:
  timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]

symbols:
  BTCUSDT:
    ccxt_symbol: "BTC/USDT:USDT"
    timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]
  ETHUSDT:
    ccxt_symbol: "ETH/USDT:USDT"
    timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]
```

### Base de Datos
Requiere conexión a PostgreSQL configurada en `config/.env`:
```env
DB_URL=postgresql+psycopg2://user:password@host:port/database
```

## 🔧 Funcionamiento Interno

### Flujo de Datos
1. **Lectura de configuración** - Símbolos y timeframes desde YAML
2. **Obtención de datos** - OHLCV desde `trading.HistoricalData`
3. **Cálculo de indicadores** - Aplicación de fórmulas técnicas
4. **Validación de tipos** - Conversión a float para estabilidad
5. **Persistencia** - Upsert en `trading.Features`

### Actualización Incremental
- Busca el último timestamp en `trading.Features`
- Obtiene solo datos nuevos desde `trading.HistoricalData`
- Calcula indicadores solo para barras nuevas
- Usa `ON CONFLICT DO UPDATE` para evitar duplicados

### Sincronización Temporal
- Calcula tiempo hasta el próximo cierre de vela
- Aplica grace period para evitar actualizaciones prematuras
- Usa jitter aleatorio para distribuir carga entre múltiples instancias

## 📊 Consultas Útiles

### Verificar Cobertura de Features
```sql
SELECT symbol, timeframe, COUNT(*) as barras,
       MIN(timestamp) as desde, MAX(timestamp) as hasta
FROM trading.Features
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;
```

### Obtener Features para ML
```sql
SELECT timestamp, rsi14, ema20, ema50, ema200, macd, atr14, bb_upper, bb_lower
FROM trading.Features
WHERE symbol = 'BTCUSDT' AND timeframe = '1m'
  AND timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp;
```

### Verificar Última Actualización
```sql
SELECT symbol, timeframe, MAX(updated_at) as ultima_actualizacion
FROM trading.Features
GROUP BY symbol, timeframe
ORDER BY ultima_actualizacion DESC;
```

## 🆘 Troubleshooting

### Problemas Comunes

**Error de tipos de datos:**
- Verificar que `HistoricalData` tenga datos correctos
- Revisar conversión a float en `fetch_candles`

**Features desactualizados:**
- Verificar que `features_updater.py` esté ejecutándose
- Revisar logs para errores de conexión a DB

**Indicadores con valores NaN:**
- Verificar que hay suficientes datos históricos (warmup period)
- Revisar configuración de períodos en indicadores

### Logs y Monitoreo
```bash
# Ver logs en tiempo real
tail -f logs/features_updater.log

# Verificar estado de la base de datos
psql -d trading_db -c "SELECT COUNT(*) FROM trading.Features;"
```

## 🔄 Integración con el Sistema

### Con Agentes ML
- Los agentes leen features desde `trading.Features`
- Usan indicadores para tomar decisiones de trading
- Requieren features actualizados para funcionar correctamente

### Con Data Layer
- Depende de `trading.HistoricalData` para datos OHLCV
- Se ejecuta después de la actualización de datos históricos
- Mantiene sincronización temporal con cierres de velas

### Con Control System
- Puede ser iniciado/detenido via comandos Telegram
- Proporciona métricas de actualización para monitoreo
- Integrado con sistema de logging del bot

## 📈 Rendimiento

### Optimizaciones
- Cálculo incremental (solo nuevas barras)
- Índices optimizados en base de datos
- Procesamiento por lotes para inserción
- Jitter para distribuir carga

### Escalabilidad
- Soporte para múltiples símbolos en paralelo
- Configuración flexible de timeframes
- Manejo eficiente de memoria con pandas
- Conexiones de DB con pool de conexiones

## 🎯 Próximas Mejoras

- [ ] Indicadores adicionales (Stochastic, Williams %R, etc.)
- [ ] Cálculo paralelo de indicadores
- [ ] Cache de indicadores calculados
- [ ] Métricas de calidad de datos
- [ ] Alertas de desactualización
