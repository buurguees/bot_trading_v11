# Mantenimiento de Base de Datos - Trading Bot

Este directorio contiene scripts para el mantenimiento y optimización de la base de datos del trading bot.

## 📁 Archivos Incluidos

### Scripts SQL
- **`db_maintenance.sql`** - Script principal de mantenimiento con índices BRIN y VACUUM/ANALYZE

### Scripts de Ejecución
- **`run_db_maintenance.bat`** - Script para Windows (Batch)
- **`run_db_maintenance.sh`** - Script para Linux/macOS (Bash)
- **`run_db_maintenance.py`** - Script en Python (multiplataforma)

### Configuración
- **`db_maintenance_config.yaml`** - Configuración personalizable del mantenimiento

## 🚀 Uso Rápido

### Windows
```cmd
cd scripts
run_db_maintenance.bat
```

### Linux/macOS
```bash
cd scripts
./run_db_maintenance.sh
```

### Python (multiplataforma)
```bash
cd scripts
python run_db_maintenance.py
```

## 🔧 Qué Hace el Mantenimiento

### 1. Índices BRIN (Block Range Indexes)
- **Propósito**: Optimizar consultas en tablas grandes con datos ordenados por timestamp
- **Tablas**: `historicaldata`, `features`, `agentpreds`, `agentsignals`, `tradeplans`, `backtests`, `backtesttrades`
- **Ventajas**: 
  - Menor uso de espacio que índices B-tree
  - Mejor rendimiento para consultas de rango temporal
  - Mantenimiento automático

### 2. Índices Compuestos
- **Propósito**: Optimizar consultas frecuentes con múltiples columnas
- **Ejemplos**: `(symbol, timeframe, timestamp)`, `(symbol, bar_ts)`
- **Ventajas**: Consultas más rápidas para filtros combinados

### 3. VACUUM ANALYZE
- **Propósito**: Limpiar espacio no utilizado y actualizar estadísticas
- **Tablas**: Todas las tablas del esquema `trading`
- **Ventajas**:
  - Libera espacio en disco
  - Actualiza estadísticas para el planificador de consultas
  - Mejora el rendimiento general

### 4. Estadísticas y Monitoreo
- **Tamaños de tablas**: Monitoreo del crecimiento de datos
- **Estadísticas de columnas**: Información para optimización
- **Rangos de fechas**: Verificación de cobertura de datos

## ⚙️ Configuración

### Variables de Entorno
```bash
export PGHOST=192.168.10.109
export PGPORT=5432
export PGDATABASE=trading_db
export PGUSER=trading_user
export PGPASSWORD=160501
```

### Personalización
Edita `db_maintenance_config.yaml` para:
- Cambiar parámetros de conexión
- Modificar tablas para índices BRIN
- Ajustar configuración de VACUUM
- Personalizar umbrales de limpieza

## 📊 Monitoreo

### Logs
- **Ubicación**: `logs/db_maintenance_YYYY-MM-DD_HH-MM-SS.log`
- **Contenido**: Salida completa de psql, errores, estadísticas
- **Rotación**: Se mantienen los últimos 10 logs

### Métricas Importantes
- **Tamaño de tablas**: Monitoreo del crecimiento
- **Tiempo de ejecución**: Duración del mantenimiento
- **Índices creados**: Confirmación de creación exitosa
- **Errores**: Detección de problemas

## ⚠️ Consideraciones Importantes

### Tiempo de Ejecución
- **Índices BRIN**: ~5-10 minutos (dependiendo del tamaño de datos)
- **VACUUM ANALYZE**: ~2-5 minutos por tabla grande
- **Total estimado**: 15-30 minutos

### Recursos del Sistema
- **CPU**: Uso moderado durante VACUUM
- **Memoria**: Incremento temporal durante ANALYZE
- **Disco**: Espacio adicional para índices (10-20% de las tablas)

### Recomendaciones
1. **Ejecutar en horarios de baja actividad**
2. **Monitorear el espacio en disco**
3. **Verificar logs después de cada ejecución**
4. **Ejecutar semanalmente o según necesidad**

## 🔍 Solución de Problemas

### Error: "psql no encontrado"
```bash
# Instalar PostgreSQL client
# Windows: Descargar desde postgresql.org
# Ubuntu/Debian: sudo apt-get install postgresql-client
# macOS: brew install postgresql
```

### Error: "Conexión rechazada"
- Verificar que PostgreSQL esté ejecutándose
- Comprobar configuración de red/firewall
- Validar credenciales en `db_maintenance_config.yaml`

### Error: "Permisos insuficientes"
- El usuario debe tener permisos de `CREATE INDEX` y `VACUUM`
- Verificar que el usuario tenga acceso al esquema `trading`

### Índices que fallan
- Algunos índices pueden fallar si ya existen
- Esto es normal y no afecta el funcionamiento
- Revisar logs para detalles específicos

## 📈 Beneficios Esperados

### Rendimiento
- **Consultas temporales**: 50-80% más rápidas
- **Consultas por símbolo**: 30-50% más rápidas
- **Consultas combinadas**: 40-60% más rápidas

### Espacio
- **Índices BRIN**: 70-90% menos espacio que B-tree
- **VACUUM**: Libera espacio no utilizado
- **Estadísticas**: Mejora la planificación de consultas

### Mantenimiento
- **Automatización**: Ejecución programada
- **Monitoreo**: Logs detallados
- **Configuración**: Fácil personalización

## 🕐 Programación Automática

### Windows (Task Scheduler)
1. Abrir "Programador de tareas"
2. Crear tarea básica
3. Configurar para ejecutar `run_db_maintenance.bat`
4. Establecer frecuencia (ej: semanal)

### Linux (Cron)
```bash
# Editar crontab
crontab -e

# Ejecutar cada domingo a las 2 AM
0 2 * * 0 /ruta/al/proyecto/scripts/run_db_maintenance.sh
```

### Python (APScheduler)
```python
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess

scheduler = BlockingScheduler()

@scheduler.scheduled_job('cron', day_of_week='sun', hour=2)
def run_maintenance():
    subprocess.run(['python', 'run_db_maintenance.py'])

scheduler.start()
```

## 📞 Soporte

Para problemas o preguntas:
1. Revisar logs en `logs/`
2. Verificar configuración en `db_maintenance_config.yaml`
3. Consultar documentación de PostgreSQL
4. Verificar permisos de usuario de base de datos
