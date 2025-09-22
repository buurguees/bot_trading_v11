# 🔧 Mantenimiento de Base de Datos - Bot Trading V11

Este directorio contiene scripts para el mantenimiento automático de la base de datos PostgreSQL del sistema de trading.

## 📁 Archivos

- **`db_maintenance.sql`** - Script SQL principal con índices BRIN y VACUUM/ANALYZE
- **`run_db_maintenance.bat`** - Script para Windows (.bat)
- **`run_db_maintenance.sh`** - Script para Linux/Mac (.sh)
- **`README.md`** - Este archivo de documentación

## 🚀 Uso Rápido

### Windows
```cmd
cd scripts\maintenance
run_db_maintenance.bat
```

### Linux/Mac
```bash
cd scripts/maintenance
chmod +x run_db_maintenance.sh
./run_db_maintenance.sh
```

## 📋 Qué hace el mantenimiento

### 1. Índices BRIN (Block Range INdexes)
Crea índices BRIN optimizados para columnas temporales en las siguientes tablas:

- **`trading.historicaldata`**
  - `timestamp` - Para consultas por rango de fechas
  - `created_at` - Para auditoría

- **`trading.trades`**
  - `entry_timestamp` - Para consultas de operaciones
  - `created_at` - Para auditoría

- **`trading.mlstrategies`**
  - `timestamp` - Para consultas de estrategias
  - `created_at` - Para auditoría

- **`trading.agentversions`**
  - `created_at` - Para versiones de modelos

- **`trading.agentpreds`** (si existe)
  - `timestamp` y `created_at`

- **`trading.agentsignals`** (si existe)
  - `timestamp` y `created_at`

- **`trading.tradeplans`** (si existe)
  - `bar_ts` y `created_at`

- **`trading.backtests`** (si existe)
  - `run_ts` y `created_at`

- **`trading.auditlog`**
  - `timestamp` - Para consultas de auditoría

### 2. VACUUM ANALYZE
Ejecuta VACUUM ANALYZE en todas las tablas principales para:
- Actualizar estadísticas del optimizador
- Liberar espacio de tuplas muertas
- Mejorar rendimiento de consultas

### 3. Estadísticas de Mantenimiento
Genera reportes sobre:
- Estado de las tablas (inserts, updates, deletes)
- Tamaño de tablas
- Índices BRIN creados
- Fechas de último VACUUM/ANALYZE

## ⚙️ Configuración

### Variables de Entorno (Linux/Mac)
```bash
export DB_HOST=192.168.10.109
export DB_PORT=5432
export DB_NAME=trading_db
export DB_USER=trading_user
export DB_PASSWORD=160501
```

### Configuración en Windows
Edita las variables en `run_db_maintenance.bat`:
```cmd
set DB_HOST=192.168.10.109
set DB_PORT=5432
set DB_NAME=trading_db
set DB_USER=trading_user
set DB_PASSWORD=160501
```

## 🔍 Monitoreo

### Verificar índices BRIN creados
```sql
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE schemaname = 'trading' 
  AND indexdef LIKE '%USING BRIN%'
ORDER BY tablename, indexname;
```

### Estado de las tablas
```sql
SELECT 
    schemaname,
    tablename,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples,
    last_vacuum,
    last_analyze
FROM pg_stat_user_tables 
WHERE schemaname = 'trading'
ORDER BY n_live_tup DESC;
```

## 📊 Beneficios de los Índices BRIN

### Ventajas
- **Tamaño pequeño**: Ocupan mucho menos espacio que B-tree
- **Rápida creación**: Se crean muy rápido incluso en tablas grandes
- **Eficientes para rangos**: Excelentes para consultas por rango de fechas
- **Mantenimiento mínimo**: No necesitan mucho mantenimiento

### Cuándo usar BRIN
- Columnas con datos ordenados (como timestamps)
- Tablas muy grandes (>1GB)
- Consultas principalmente por rangos
- Cuando el espacio es limitado

### Cuándo NO usar BRIN
- Consultas por valores específicos (no rangos)
- Datos desordenados o aleatorios
- Tablas pequeñas (<100MB)
- Actualizaciones muy frecuentes

## 🛠️ Configuración Avanzada de PostgreSQL

### Optimizar BRIN
```sql
-- Ajustar páginas por rango BRIN (default: 128)
ALTER SYSTEM SET brin_page_items_per_range = 64;
```

### Configurar autovacuum para tablas grandes
```sql
-- HistoricalData (tabla más grande)
ALTER TABLE trading.historicaldata 
SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE trading.historicaldata 
SET (autovacuum_analyze_scale_factor = 0.05);
```

### Aplicar configuraciones
```sql
SELECT pg_reload_conf();
```

## 📝 Logs

Los scripts generan logs detallados:
- **Windows**: Mensajes en consola + opción de abrir log
- **Linux/Mac**: Archivo `db_maintenance_YYYYMMDD_HHMMSS.log`

## ⚠️ Consideraciones

1. **Tiempo de ejecución**: Los índices BRIN se crean con `CONCURRENTLY` para no bloquear la tabla
2. **Espacio en disco**: VACUUM puede requerir espacio temporal adicional
3. **Horario recomendado**: Ejecutar durante horarios de baja actividad
4. **Frecuencia**: Ejecutar semanalmente o cuando el rendimiento disminuya

## 🔧 Solución de Problemas

### Error de conexión
- Verificar que PostgreSQL esté ejecutándose
- Comprobar credenciales y configuración de red
- Verificar que `psql` esté en el PATH

### Error de permisos
- Verificar que el usuario tenga permisos de DDL
- Ejecutar como superusuario si es necesario

### Índices no se crean
- Verificar que las tablas existan
- Comprobar que no haya índices duplicados
- Revisar logs para errores específicos

## 📞 Soporte

Para problemas o dudas:
1. Revisar los logs generados
2. Verificar la configuración de PostgreSQL
3. Consultar la documentación de PostgreSQL sobre BRIN
4. Revisar el estado de las tablas con las consultas de monitoreo
