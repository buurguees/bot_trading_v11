# Sistema de Procesamiento de Señales

Este módulo se encarga de convertir predicciones de `trading.agentpreds` en señales de trading para `trading.agentsignals`.

## 🏗️ Arquitectura

```
agentpreds (predicciones) → signal_processor.py → agentsignals (señales)
```

## 📁 Estructura de Archivos

```
core/ml/signals/
├── signal_processor.py          # Procesador principal
└── README.md                   # Este archivo

config/signals/
└── signal_processing.yaml      # Configuración

scripts/
├── run_signal_processor.py     # Script de ejecución
└── start_signal_processor_daemon.py  # Daemon para ejecución continua
```

## ⚙️ Configuración

### Archivo: `config/signals/signal_processing.yaml`

```yaml
signal_processing:
  batch_size: 1000
  lookback_minutes: 5
  min_confidence: 0.6
  min_strength: 0.1
  side_threshold: 0.5
  
  filters:
    - duplicate_filter
    - time_filter
    - confidence_filter
    - strength_filter
  
  strength_calculation: "abs(prob - 0.5)"
  
  realtime:
    enabled: true
    interval_seconds: 30
    
  batch:
    enabled: true
    schedule: "0 */5 * * *"
```

## 🚀 Uso

### 1. Ejecución en Tiempo Real (Una vez)

```bash
python scripts/run_signal_processor.py --mode realtime
```

### 2. Ejecución Continua (Daemon)

```bash
python scripts/start_signal_processor_daemon.py --interval 30
```

### 3. Procesamiento en Lote

```bash
python scripts/run_signal_processor.py --mode batch \
  --start-time "2024-01-01 00:00:00" \
  --end-time "2024-01-01 23:59:59"
```

### 4. Procesamiento de Símbolo Específico

```bash
python scripts/run_signal_processor.py --mode realtime --symbol BTCUSDT
```

## 🔧 Funciones Principales

### `SignalProcessor`

- **`get_recent_predictions()`**: Obtiene predicciones recientes de `agentpreds`
- **`convert_prediction_to_signal()`**: Convierte predicción a señal
- **`apply_signal_filters()`**: Aplica filtros a las señales
- **`save_signals_to_db()`**: Guarda señales en `agentsignals`
- **`process_predictions_to_signals()`**: Función principal de procesamiento

### Filtros Disponibles

1. **`duplicate_filter`**: Elimina señales duplicadas
2. **`time_filter`**: Filtra señales muy cercanas en el tiempo
3. **`confidence_filter`**: Filtra por confianza mínima
4. **`strength_filter`**: Filtra por fuerza mínima

## 📊 Métricas

El procesador genera métricas detalladas:

```python
{
    "predictions": 100,      # Predicciones procesadas
    "signals": 85,           # Señales generadas
    "filtered_signals": 80,  # Señales después de filtros
    "saved_signals": 80,     # Señales guardadas en DB
    "errors": 0,             # Errores encontrados
    "processing_time": 1.23, # Tiempo de procesamiento (s)
    "conversion_rate": 0.85, # Tasa de conversión
    "filter_rate": 0.94      # Tasa de filtrado
}
```

## 🔄 Flujo de Datos

1. **Lectura**: Obtiene predicciones de `trading.agentpreds`
2. **Conversión**: Convierte probabilidades a señales direccionales
3. **Filtrado**: Aplica filtros de calidad
4. **Guardado**: Inserta señales en `trading.agentsignals`
5. **Seguimiento**: Marca predicciones como procesadas

## ⚠️ Consideraciones

- **Evita duplicados**: Usa un set de IDs procesados
- **Manejo de errores**: Continúa procesando aunque falle una predicción
- **Performance**: Procesa en lotes para eficiencia
- **Configurabilidad**: Todas las reglas son configurables

## 🐛 Debugging

### Logs

Los logs se guardan en:
- `logs/signal_processor.log` - Logs del procesador
- `logs/signal_processor_daemon.log` - Logs del daemon

### Verificar Señales Generadas

```sql
SELECT symbol, timeframe, timestamp, side, strength, 
       (meta->>'confidence')::float as confidence
FROM trading.agentsignals 
WHERE created_at >= NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;
```

### Verificar Predicciones Procesadas

```sql
SELECT COUNT(*) as total_predicciones,
       COUNT(DISTINCT symbol) as simbolos,
       MIN(created_at) as primera,
       MAX(created_at) as ultima
FROM trading.agentpreds 
WHERE created_at >= NOW() - INTERVAL '1 hour';
```
