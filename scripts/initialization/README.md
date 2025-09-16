# Inicialización de Base de Datos - Bot Trading v11

Este directorio contiene los scripts necesarios para inicializar y configurar la base de datos PostgreSQL del sistema de trading.

## 📁 Archivos Incluidos

- **`init_db.sql`** - Script SQL completo para crear la base de datos y tablas
- **`init_db.py`** - Script Python para automatizar la inicialización
- **`verify_db.py`** - Script de verificación para comprobar la configuración
- **`README.md`** - Esta documentación

## 🚀 Inicialización Rápida

### Opción 1: Script Python (Recomendado)

```bash
# Activar entorno virtual
.\venv\Scripts\activate

# Ejecutar inicialización
python scripts/initialization/init_db.py

# Verificar configuración
python scripts/initialization/verify_db.py
```

### Opción 2: Script SQL Directo

```bash
# Conectar a PostgreSQL como superusuario
psql -U postgres

# Ejecutar script SQL
\i scripts/initialization/init_db.sql

# Verificar tablas
\c trading_db
\dt trading.*
```

## 📊 Estructura de la Base de Datos

### Esquema `trading`

#### Tabla `HistoricalData`
- **Propósito**: Almacena datos OHLCV históricos
- **Particiones**: Por año (2025, 2026, etc.)
- **Índices**: Optimizados para consultas por símbolo, timeframe y timestamp
- **Campos**:
  - `symbol` (VARCHAR): Símbolo del activo (ej: BTCUSDT)
  - `timeframe` (VARCHAR): Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
  - `timestamp` (TIMESTAMP WITH TIME ZONE): Timestamp con zona horaria
  - `open`, `high`, `low`, `close` (NUMERIC): Precios OHLC
  - `volume` (NUMERIC): Volumen de trading

#### Tabla `Trades`
- **Propósito**: Registra operaciones de trading
- **Campos**:
  - `symbol` (VARCHAR): Símbolo del activo
  - `side` (VARCHAR): Lado de la operación (long/short)
  - `quantity` (NUMERIC): Cantidad
  - `price` (NUMERIC): Precio de ejecución
  - `pnl` (NUMERIC): Profit and Loss
  - `leverage` (NUMERIC): Leverage utilizado
  - `entry_timestamp`, `exit_timestamp` (TIMESTAMP): Timestamps de entrada y salida

#### Tabla `MLStrategies`
- **Propósito**: Almacena estrategias de Machine Learning
- **Campos**:
  - `symbol` (VARCHAR): Símbolo del activo
  - `action` (VARCHAR): Acción predicha (long/short/hold)
  - `timeframes` (JSONB): Timeframes utilizados
  - `indicators` (JSONB): Indicadores técnicos
  - `tools` (JSONB): Herramientas adicionales
  - `pnl` (NUMERIC): Resultado de la estrategia
  - `performance` (NUMERIC): Métrica de rendimiento (ej: Sharpe ratio)
  - `confidence_score` (NUMERIC): Nivel de confianza (0-1)
  - `outcome` (VARCHAR): Resultado (win/loss/neutral)

#### Tabla `AuditLog`
- **Propósito**: Registro de auditoría para cambios en las tablas
- **Campos**:
  - `table_name` (VARCHAR): Tabla modificada
  - `action` (VARCHAR): Tipo de acción (INSERT/UPDATE/DELETE)
  - `record_id` (BIGINT): ID del registro afectado
  - `changed_by` (VARCHAR): Usuario que realizó el cambio
  - `timestamp` (TIMESTAMP): Momento del cambio
  - `details` (JSONB): Detalles adicionales

## ⚙️ Configuración

### Variables de Entorno

Asegúrate de que `config/.env` contenga:

```env
DB_URL=postgresql+psycopg2://trading_user:160501@192.168.10.109:5432/trading_db
DB_HOST=192.168.10.109
DB_PORT=5432
DB_NAME=trading_db
DB_USER=trading_user
DB_PASSWORD=160501
```

### Usuario de Base de Datos

El script crea un usuario `trading_user` con contraseña `160501` y permisos completos sobre la base de datos `trading_db`.

**⚠️ Importante**: Cambia la contraseña en producción.

## 🔍 Verificación

### Comandos de Verificación

```sql
-- Conectar a la base de datos
psql -U trading_user -d trading_db

-- Ver todas las tablas
\dt trading.*

-- Ver estructura de una tabla
\d trading.HistoricalData

-- Ver índices
\di trading.*

-- Ver particiones
SELECT schemaname, tablename FROM pg_tables WHERE tablename LIKE 'HistoricalData_%';

-- Consultar datos de muestra
SELECT * FROM trading.HistoricalData LIMIT 5;
SELECT * FROM trading.MLStrategies LIMIT 5;
```

### Script de Verificación Automática

```bash
python scripts/initialization/verify_db.py
```

Este script verifica:
- ✅ Conexión a la base de datos
- ✅ Existencia de todas las tablas
- ✅ Creación de índices
- ✅ Datos de muestra
- ✅ Consultas básicas

## 🛠️ Solución de Problemas

### Error de Conexión

```bash
# Verificar que PostgreSQL esté ejecutándose
pg_ctl status

# Verificar configuración de conexión
psql -U postgres -c "SELECT version();"
```

### Error de Permisos

```bash
# Conectar como superusuario y otorgar permisos
psql -U postgres -d trading_db
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trading_user;
GRANT USAGE ON SCHEMA trading TO trading_user;
```

### Error de Particiones

```sql
-- Crear partición manualmente si es necesario
CREATE TABLE trading.HistoricalData_2025 PARTITION OF trading.HistoricalData
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

## 📈 Optimizaciones

### Particionado
- Las tablas `HistoricalData` están particionadas por año
- Mejora el rendimiento en consultas históricas
- Facilita el mantenimiento de datos antiguos

### Índices
- Índices compuestos para consultas comunes
- Índices de cobertura para consultas específicas
- Índices en campos de timestamp para ordenamiento

### Configuración PostgreSQL
Para mejor rendimiento, considera ajustar:

```sql
-- En postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

## 🔄 Mantenimiento

### Limpieza de Datos Antiguos

```sql
-- Eliminar datos históricos antiguos (ej: > 2 años)
DELETE FROM trading.HistoricalData 
WHERE timestamp < NOW() - INTERVAL '2 years';
```

### Actualización de Estadísticas

```sql
-- Actualizar estadísticas para optimizador
ANALYZE trading.HistoricalData;
ANALYZE trading.Trades;
ANALYZE trading.MLStrategies;
```

### Backup

```bash
# Backup completo
pg_dump -U trading_user -d trading_db > backup_$(date +%Y%m%d).sql

# Backup solo esquema
pg_dump -U trading_user -d trading_db --schema-only > schema_backup.sql
```

## 📚 Referencias

- [Documentación PostgreSQL](https://www.postgresql.org/docs/)
- [Particionado en PostgreSQL](https://www.postgresql.org/docs/current/ddl-partitioning.html)
- [Índices en PostgreSQL](https://www.postgresql.org/docs/current/indexes.html)
