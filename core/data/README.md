# Data Layer & Base de Datos

Este módulo gestiona todo el ciclo de vida del dato: descarga histórica (Bitget, Futuros), ingesta incremental, actualización en tiempo real, alineación multi-TF y acceso estructurado a PostgreSQL.

## 🧭 Visión general

```
config/trading/symbols.yaml  ──▶  trading.Symbols (catálogo FUTUROS)
                                  │
historical_downloader.py ───────▶ trading.HistoricalData (OHLCV)
realtime_fetcher.py ────────────▶ trading.HistoricalData (append)
timeframe_aligner.py ───────────▶ CSV/frames alineados para ML
database.py ────────────────────▶ ORM/consultas (Trades, MLStrategies, AuditLog)
```

## 📁 Archivos clave

### `database.py`
ORM/engine de PostgreSQL (SQLAlchemy). Define tablas y utilidades:

- **trading.HistoricalData** — OHLCV (Futuros).
- **trading.Trades** — operaciones del bot (entry/exit, pnl, leverage).
- **trading.MLStrategies** — setups/estrategias evaluadas.
- **trading.AuditLog** — auditoría de cambios.

**Helpers:** `get_historical_data`, `insert_trade`, `insert_strategy`, `execute_query`, etc.

### `historical_downloader.py`
Descarga histórico completo por símbolo/TF desde Bitget (Futuros) y hace upsert idempotente en HistoricalData. Reanuda desde el último timestamp existente.

### `realtime_fetcher.py` (opcional si lo usas ahora o más adelante)
Captura velas recientes (REST/WebSocket) y mantiene la DB actualizada.

### `timeframe_aligner.py`
Alinea series multi-TF por timestamp para entrenamiento/evaluación (salida CSV o DataFrame).

### `data_updater.py`
Lógica común de append seguro (sin duplicados), validaciones y pequeñas reparaciones de huecos.

## 🗄️ Esquema de la DB

**Esquema:** `trading` (owner: `trading_user`)

### 1) `trading.Symbols` (FUTUROS únicamente)

Catálogo de símbolos y metadatos operativos.

| columna | tipo | notas |
|---------|------|-------|
| id | BIGSERIAL (PK) | |
| symbol | VARCHAR(20) | BTCUSDT, ETHUSDT, … (UNIQUE) |
| base_asset | VARCHAR(20) | BTC |
| quote_asset | VARCHAR(20) | USDT |
| market | VARCHAR(10) | 'futures' (CHECK market='futures') |
| tfs | VARCHAR(5)[] | {'1m','5m','15m','1h','4h','1d'} |
| tick_size | NUMERIC | filtros exchange |
| step_size | NUMERIC | filtros exchange |
| min_qty | NUMERIC | filtros exchange |
| leverage_min | INTEGER | p.ej. 1 |
| leverage_max | INTEGER | p.ej. 50 / 80 / 125 |
| filters | JSONB | extra |
| is_active | BOOLEAN | |

**Índices:** `idx_symbols_active`, `idx_symbols_leverage`.

**Semillado:** se carga desde `config/trading/symbols.yaml` (se filtra/impone futures).

### 2) `trading.HistoricalData`

OHLCV por (symbol, timeframe, timestamp) en UTC.

| columna | tipo |
|---------|------|
| id | BIGSERIAL (PK) |
| symbol | VARCHAR |
| timeframe | VARCHAR(5) |
| timestamp | TIMESTAMPTZ |
| open | NUMERIC(15,8) |
| high | NUMERIC(15,8) |
| low | NUMERIC(15,8) |
| close | NUMERIC(15,8) |
| volume | NUMERIC(20,8) |
| created_at | TIMESTAMPTZ (default now) |
| updated_at | TIMESTAMPTZ (default now) |

**Constraint:** `UNIQUE(symbol, timeframe, timestamp)`  
**Índice:** `idx_hist_symbol_tf_ts(symbol, timeframe, timestamp DESC)`

Esto permite upsert idempotente y consultas rápidas por rango de fechas.

### 3) `trading.Trades`

Operaciones ejecutadas por el bot (paper/live). Guarda side, qty, price, leverage, PnL, entry/exit.

### 4) `trading.MLStrategies`

Registro de setups/estrategias (acción, indicadores, TFs, score, performance, outcome). Útil para memoria de estrategias.

### 5) `trading.AuditLog`

Auditoría básica (tabla afectada, acción, quién/cuándo, detalle JSONB).

## 🔧 Variables de entorno

En tu `.env`:

```bash
DB_URL=postgresql+psycopg2://trading_user:TU_PASSWORD@192.168.10.109:5432/trading_db
# Claves de exchange/telegram fuera de este repo y nunca en git.
```

Revisa que `pg_hba.conf` permita tu red y `postgresql.conf` escuche en tu IP.

## 🚀 Operativa: de 0 a histórico completo

### Semillar símbolos (solo Futuros)

1. Prepara `config/trading/symbols.yaml` con `market: futures` y `tfs` por símbolo.
2. Ejecuta tu `seed_futures_symbols.py` (o el semillado integrado si ya lo tienes).

### Crear constraints + índice (si aún no están):

```sql
ALTER TABLE trading.HistoricalData
  ADD CONSTRAINT IF NOT EXISTS unique_ohlcv UNIQUE(symbol, timeframe, timestamp);

CREATE INDEX IF NOT EXISTS idx_hist_symbol_tf_ts
  ON trading.HistoricalData(symbol, timeframe, timestamp DESC);
```

### Descargar histórico (todos los símbolos/TFs del YAML):

```bash
python core/data/historical_downloader.py
```

- Reanuda automáticamente desde el último timestamp que encuentre en DB.
- Inserta por lotes y usa `ON CONFLICT DO NOTHING` para evitar duplicados.

### (Opcional) Tiempo real

```bash
python core/data/realtime_fetcher.py
```

Mantiene la DB al día; útil para backtests cercanos al presente y paper/live.

### Alinear multi-TF (para ML)

```bash
python core/data/timeframe_aligner.py --symbol BTCUSDT --base_tf 1m --with 5m,15m,1h,4h,1d
```

Genera CSV/frames con columnas multi-TF alineadas en el mismo timestamp.

## 📚 Consultas útiles

### Cobertura por símbolo/TF:

```sql
SELECT symbol, timeframe, COUNT(*) velas,
       MIN(timestamp) desde, MAX(timestamp) hasta
FROM trading.HistoricalData
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;
```

### Serie para entrenar:

```sql
SELECT timestamp, open, high, low, close, volume
FROM trading.HistoricalData
WHERE symbol='BTCUSDT' AND timeframe='1m'
  AND timestamp BETWEEN NOW() - INTERVAL '365 days' AND NOW()
ORDER BY timestamp;
```

### Último timestamp disponible (para una reanudación controlada):

```sql
SELECT MAX(timestamp) FROM trading.HistoricalData
WHERE symbol='BTCUSDT' AND timeframe='1m';
```

## ⚡ Rendimiento & Escalado

- Batch inserts en el downloader, UNIQUE + índice compuesto para lecturas rápidas.
- Alineación multi-TF fuera de la DB (en Python) para evitar JOINs costosos.
- Particionado por año (recomendado a partir de decenas de millones de filas):

```sql
-- Ejemplo: convertir a particionada por rango (timestamp) y crear particiones anuales.
-- (Planificar con cuidado si la tabla ya tiene datos)
```

Si lo quieres, te dejo script listo para migrar sin downtime.

## 🧩 Integración con el bot

- Los agentes leen de `HistoricalData` y escriben trades en `Trades`.
- Las estrategias evaluadas se guardan en `MLStrategies` (top/bottom-1000 para memoria).
- `AuditLog` registra acciones de mantenimiento/migración.

## 🆘 Troubleshooting rápido

- **Permisos:** asegura que `trading_user` es owner del esquema/objetos y tiene USAGE, CREATE.
- **Conexión LAN:** añade regla de firewall Windows para TCP/5432 y entrada en `pg_hba.conf`.
- **Duplicados:** verifica que existe `UNIQUE(symbol,timeframe,timestamp)`.
- **Velocidad:** crea el índice `idx_hist_symbol_tf_ts` y usa filtros por rango de fechas.

## TL;DR

- Un solo repositorio de OHLCV (`trading.HistoricalData`) con clave compuesta e índice.
- Semillado de símbolos Futuros desde YAML.
- Downloader reanudable e idempotente.
- Alineación multi-TF y utilidades ORM listas para ML/trading.
