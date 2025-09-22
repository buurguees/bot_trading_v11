# ⚡ INFERENCIA EN TIEMPO REAL DE ALTA PERFORMANCE

## 📋 **DESCRIPCIÓN**

Sistema de inferencia en tiempo real optimizado para trading con arquitectura de baja latencia, manejo de múltiples modelos, streaming de datos y monitoreo de performance.

## ✨ **CARACTERÍSTICAS PRINCIPALES**

### **1. ARQUITECTURA DE BAJA LATENCIA**
- **Pool de modelos pre-cargados** en memoria
- **Cache de features** calculadas recientemente
- **Predicciones asíncronas** con asyncio
- **Procesamiento en lotes** para múltiples símbolos

### **2. MANEJO DE MÚLTIPLES MODELOS**
- **Carga lazy** de modelos por símbolo/timeframe
- **Versionado automático**: cargar último modelo promovido
- **Fallback a modelo anterior** si el nuevo falla
- **Pool inteligente** con evicción LRU

### **3. STREAMING DE DATOS**
- **Conexión persistente** a DB con pooling
- **Trigger en nuevas barras** OHLCV
- **Batch processing** de múltiples símbolos
- **Cache de features** con TTL configurable

### **4. MONITOREO DE LATENCIA**
- **Métricas de tiempo** por etapa: carga, features, predicción
- **Alertas si latencia > 5 segundos**
- **Logging de performance** por símbolo
- **Dashboard de métricas** en tiempo real

### **5. SISTEMA DE SALUD**
- **Health checks** cada minuto
- **Auto-restart** en caso de memory leaks
- **Métricas de CPU/memoria** del proceso
- **Alertas automáticas** de recursos

### **6. CONFIGURACIÓN ADAPTATIVA**
- **Ajuste automático** de batch size según latencia
- **Escalado de workers** según carga
- **Priorización por volumen** de trading
- **Configuración dinámica** de parámetros

## 🛠️ **INSTALACIÓN**

### **Dependencias adicionales:**
```bash
pip install asyncio psutil aiofiles aiohttp
```

### **Configuración inicial:**
```python
from core.ml.inference.infer_realtime import create_inference_engine

# Crear motor con configuración personalizada
config = {
    'max_latency_ms': 3000,  # 3 segundos máximo
    'health_check_interval': 30,  # 30 segundos
    'cache_ttl_seconds': 180,  # 3 minutos
    'max_memory_mb': 1024,  # 1GB
    'model_pool_size': 20,
    'feature_cache_size': 500
}

engine = create_inference_engine(config)
```

## 🚀 **USO**

### **INICIO BÁSICO:**
```python
import asyncio
from core.ml.inference.infer_realtime import start_realtime_inference

# Iniciar sistema completo
asyncio.run(start_realtime_inference())
```

### **USO AVANZADO:**
```python
import asyncio
from core.ml.inference.infer_realtime import create_inference_engine

async def main():
    # Crear motor
    engine = create_inference_engine()
    
    # Iniciar en background
    inference_task = asyncio.create_task(engine.start())
    
    # Esperar inicialización
    await asyncio.sleep(2)
    
    # Solicitar predicciones
    await engine.predict("BTCUSDT", "1m", 1)
    await engine.predict("ETHUSDT", "5m", 3)
    await engine.predict("ADAUSDT", "1h", 5)
    
    # Obtener métricas
    metrics = engine.get_metrics()
    print(f"Métricas: {metrics}")
    
    # Cerrar
    inference_task.cancel()
    await inference_task

asyncio.run(main())
```

### **PREDICCIONES EN LOTE:**
```python
async def batch_predictions(engine, symbols, timeframes, horizons):
    """Realizar predicciones en lote"""
    tasks = []
    
    for symbol in symbols:
        for tf in timeframes:
            for horizon in horizons:
                task = engine.predict(symbol, tf, horizon)
                tasks.append(task)
    
    # Ejecutar todas las predicciones en paralelo
    await asyncio.gather(*tasks)
```

## ⚙️ **CONFIGURACIÓN**

### **PARÁMETROS PRINCIPALES:**
```python
PERFORMANCE_CONFIG = {
    'max_latency_ms': 5000,        # Latencia máxima permitida
    'health_check_interval': 60,    # Intervalo de health check (segundos)
    'cache_ttl_seconds': 300,      # TTL del cache de features
    'max_memory_mb': 2048,         # Límite de memoria (MB)
    'batch_size_initial': 10,      # Tamaño inicial de lote
    'batch_size_max': 100,         # Tamaño máximo de lote
    'workers_initial': 2,          # Workers iniciales
    'workers_max': 8,              # Workers máximos
    'model_pool_size': 50,         # Tamaño del pool de modelos
    'feature_cache_size': 1000     # Tamaño del cache de features
}
```

### **CONFIGURACIÓN DE BASE DE DATOS:**
```python
# Pool de conexiones optimizado
DB_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_pre_ping': True,
    'pool_recycle': 3600,
    'echo': False
}
```

## 📊 **MONITOREO Y MÉTRICAS**

### **MÉTRICAS DE PERFORMANCE:**
```python
metrics = engine.get_metrics()

print(f"Predicciones totales: {metrics['performance']['total_predictions']}")
print(f"Latencia promedio: {metrics['performance']['avg_latency_ms']:.1f}ms")
print(f"Latencia máxima: {metrics['performance']['max_latency_ms']:.1f}ms")
print(f"Tasa de error: {metrics['performance']['error_rate']:.1f}%")
print(f"Hit rate cache: {metrics['performance']['cache_hit_rate']:.1f}%")
```

### **MÉTRICAS DEL SISTEMA:**
```python
print(f"Memoria: {metrics['system']['memory_usage_mb']:.1f}MB")
print(f"CPU: {metrics['system']['cpu_usage_percent']:.1f}%")
print(f"Modelos cargados: {metrics['system']['models_loaded']}")
print(f"Último health check: {metrics['system']['last_health_check']}")
```

### **MÉTRICAS DEL POOL DE MODELOS:**
```python
pool_stats = metrics['model_pool']
print(f"Modelos en pool: {pool_stats['total_models']}/{pool_stats['max_size']}")

for model_key, model_info in pool_stats['models'].items():
    print(f"  {model_key}: {model_info['predictions_count']} pred, "
          f"{model_info['avg_latency_ms']:.1f}ms avg")
```

### **MÉTRICAS DEL CACHE:**
```python
cache_stats = metrics['feature_cache']
print(f"Entradas en cache: {cache_stats['total_entries']}")
print(f"Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
print(f"TTL: {cache_stats['ttl_seconds']}s")
```

## 🔧 **ARQUITECTURA INTERNA**

### **COMPONENTES PRINCIPALES:**

1. **RealtimeInferenceEngine**: Motor principal
2. **ModelPool**: Pool de modelos pre-cargados
3. **FeatureCacheManager**: Cache de features calculadas
4. **PerformanceMetrics**: Métricas del sistema

### **FLUJO DE PREDICCIÓN:**

1. **Solicitud** → `predict()` → Cola de predicciones
2. **Procesador** → Obtener modelo del pool
3. **Features** → Verificar cache → Calcular si necesario
4. **Predicción** → Modelo + Features → Resultado
5. **Resultado** → Guardar en DB → Logging

### **TAREAS ASÍNCRONAS:**

- `_health_check_loop()`: Monitoreo de salud
- `_prediction_processor()`: Procesamiento de predicciones
- `_result_processor()`: Manejo de resultados
- `_metrics_collector()`: Recolección de métricas

## 🧪 **PRUEBAS**

### **EJECUTAR PRUEBAS:**
```bash
python test_realtime_inference.py
```

### **PRUEBAS INCLUIDAS:**
1. **Inferencia Básica**: Carga y predicción básica
2. **Métricas de Performance**: Medición de latencia
3. **Funcionalidad de Cache**: Verificación de cache hit/miss
4. **Monitoreo de Salud**: Health checks y alertas
5. **Manejo de Errores**: Robustez ante errores

### **PRUEBA MANUAL:**
```python
import asyncio
from core.ml.inference.infer_realtime import create_inference_engine

async def test_manual():
    engine = create_inference_engine()
    
    # Iniciar
    task = asyncio.create_task(engine.start())
    await asyncio.sleep(2)
    
    # Probar predicciones
    await engine.predict("BTCUSDT", "1m", 1)
    await asyncio.sleep(1)
    
    # Ver métricas
    metrics = engine.get_metrics()
    print(metrics)
    
    # Cerrar
    task.cancel()
    await task

asyncio.run(test_manual())
```

## 🚨 **SOLUCIÓN DE PROBLEMAS**

### **LATENCIA ALTA:**
```python
# Reducir batch size
config['batch_size_initial'] = 5
config['batch_size_max'] = 20

# Aumentar workers
config['workers_initial'] = 4
config['workers_max'] = 12

# Reducir TTL del cache
config['cache_ttl_seconds'] = 60
```

### **MEMORIA ALTA:**
```python
# Reducir pool de modelos
config['model_pool_size'] = 10

# Reducir cache de features
config['feature_cache_size'] = 200

# Reducir límite de memoria
config['max_memory_mb'] = 512
```

### **ERRORES DE MODELO:**
```python
# Verificar que hay modelos promovidos
# Verificar rutas de artefactos
# Verificar permisos de archivos
```

### **CACHE NO FUNCIONA:**
```python
# Verificar TTL
config['cache_ttl_seconds'] = 300

# Verificar tamaño
config['feature_cache_size'] = 1000

# Verificar logs de cache
```

## 📈 **RENDIMIENTO ESPERADO**

### **LATENCIA:**
- **Predicción simple**: 10-50ms
- **Con cache hit**: 5-20ms
- **Con cache miss**: 50-200ms
- **Múltiples símbolos**: 100-500ms

### **THROUGHPUT:**
- **Predicciones por segundo**: 100-1000
- **Símbolos simultáneos**: 10-50
- **Timeframes simultáneos**: 5-20

### **RECURSOS:**
- **Memoria**: 100-500MB
- **CPU**: 10-50%
- **Conexiones DB**: 5-20

## 🎯 **BENEFICIOS**

1. **Baja Latencia**: Predicciones en milisegundos
2. **Alta Disponibilidad**: Sistema robusto y resiliente
3. **Escalabilidad**: Maneja múltiples símbolos/timeframes
4. **Eficiencia**: Cache inteligente y pooling
5. **Monitoreo**: Métricas completas en tiempo real
6. **Flexibilidad**: Configuración adaptativa

## 📁 **ARCHIVOS GENERADOS**

### **LOGS:**
- `logs/infer_realtime.log` - Log principal
- Logs de performance por símbolo
- Alertas de latencia y recursos

### **CACHE:**
- Cache de modelos en memoria
- Cache de features en memoria
- Estadísticas de hit/miss

### **BASE DE DATOS:**
- `trading.agentpreds` - Predicciones guardadas
- Métricas de performance
- Logs de errores

## 🔄 **INTEGRACIÓN**

### **CON SISTEMA DE TRADING:**
```python
# Integrar con sistema de trading
from core.ml.inference.infer_realtime import create_inference_engine

class TradingSystem:
    def __init__(self):
        self.inference_engine = create_inference_engine()
    
    async def start(self):
        # Iniciar inferencia
        self.inference_task = asyncio.create_task(
            self.inference_engine.start()
        )
    
    async def get_prediction(self, symbol, timeframe, horizon):
        # Solicitar predicción
        await self.inference_engine.predict(symbol, timeframe, horizon)
        
        # Procesar resultado (implementar callback)
        pass
```

### **CON DASHBOARD:**
```python
# Integrar con dashboard web
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/metrics')
def get_metrics():
    return jsonify(engine.get_metrics())

@app.route('/api/predict')
def predict():
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe')
    horizon = int(request.args.get('horizon', 1))
    
    asyncio.create_task(
        engine.predict(symbol, timeframe, horizon)
    )
    
    return jsonify({'status': 'prediction_requested'})
```

¡El sistema de inferencia en tiempo real está listo para trading de alta frecuencia! ⚡🚀📊
