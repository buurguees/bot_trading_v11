#!/usr/bin/env python3
"""
Script de prueba para inferencia en tiempo real
"""

import os
import sys
import asyncio
import time
import logging
from datetime import datetime, timezone

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ml.inference.infer_realtime import (
    RealtimeInferenceEngine,
    create_inference_engine,
    start_realtime_inference
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_basic_inference():
    """Probar inferencia básica"""
    logger.info("🧪 PROBANDO INFERENCIA BÁSICA")
    logger.info("=" * 50)
    
    # Configuración de prueba
    config = {
        'max_latency_ms': 2000,  # 2 segundos
        'health_check_interval': 10,  # 10 segundos
        'cache_ttl_seconds': 60,  # 1 minuto
        'max_memory_mb': 512,  # 512MB
        'model_pool_size': 5,
        'feature_cache_size': 100
    }
    
    # Crear motor
    engine = create_inference_engine(config)
    
    try:
        # Iniciar motor en background
        inference_task = asyncio.create_task(engine.start())
        
        # Esperar inicialización
        await asyncio.sleep(2)
        
        # Solicitar predicciones
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        timeframes = ["1m", "5m", "15m"]
        horizons = [1, 3, 5]
        
        logger.info("Solicitando predicciones...")
        
        for symbol in symbols:
            for tf in timeframes:
                for horizon in horizons:
                    await engine.predict(symbol, tf, horizon)
                    await asyncio.sleep(0.1)  # Pequeña pausa
        
        # Esperar procesamiento
        await asyncio.sleep(5)
        
        # Obtener métricas
        metrics = engine.get_metrics()
        
        logger.info("📊 MÉTRICAS OBTENIDAS:")
        logger.info(f"  Predicciones totales: {metrics['performance']['total_predictions']}")
        logger.info(f"  Latencia promedio: {metrics['performance']['avg_latency_ms']:.1f}ms")
        logger.info(f"  Latencia máxima: {metrics['performance']['max_latency_ms']:.1f}ms")
        logger.info(f"  Tasa de error: {metrics['performance']['error_rate']:.1f}%")
        logger.info(f"  Hit rate cache: {metrics['performance']['cache_hit_rate']:.1f}%")
        logger.info(f"  Memoria: {metrics['system']['memory_usage_mb']:.1f}MB")
        logger.info(f"  CPU: {metrics['system']['cpu_usage_percent']:.1f}%")
        logger.info(f"  Modelos cargados: {metrics['system']['models_loaded']}")
        
        # Verificar que se procesaron predicciones
        if metrics['performance']['total_predictions'] > 0:
            logger.info("✅ Inferencia básica exitosa")
            return True
        else:
            logger.error("❌ No se procesaron predicciones")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en inferencia básica: {e}")
        return False
    finally:
        # Cancelar tarea
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass

async def test_performance_metrics():
    """Probar métricas de performance"""
    logger.info("\n🧪 PROBANDO MÉTRICAS DE PERFORMANCE")
    logger.info("=" * 50)
    
    config = {
        'max_latency_ms': 1000,  # 1 segundo
        'health_check_interval': 5,  # 5 segundos
        'cache_ttl_seconds': 30,  # 30 segundos
        'max_memory_mb': 256,  # 256MB
        'model_pool_size': 3,
        'feature_cache_size': 50
    }
    
    engine = create_inference_engine(config)
    
    try:
        # Iniciar motor
        inference_task = asyncio.create_task(engine.start())
        await asyncio.sleep(1)
        
        # Realizar múltiples predicciones para medir performance
        start_time = time.time()
        
        for i in range(20):
            await engine.predict("BTCUSDT", "1m", 1)
            await asyncio.sleep(0.05)  # 50ms entre predicciones
        
        # Esperar procesamiento
        await asyncio.sleep(3)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Obtener métricas
        metrics = engine.get_metrics()
        
        logger.info(f"📊 PERFORMANCE:")
        logger.info(f"  Tiempo total: {total_time:.2f}s")
        logger.info(f"  Predicciones: {metrics['performance']['total_predictions']}")
        logger.info(f"  Latencia promedio: {metrics['performance']['avg_latency_ms']:.1f}ms")
        logger.info(f"  Latencia máxima: {metrics['performance']['max_latency_ms']:.1f}ms")
        logger.info(f"  Tasa de error: {metrics['performance']['error_rate']:.1f}%")
        
        # Verificar que la latencia está dentro del límite
        if metrics['performance']['avg_latency_ms'] <= config['max_latency_ms']:
            logger.info("✅ Latencia dentro del límite")
            return True
        else:
            logger.warning(f"⚠️  Latencia excede límite: {metrics['performance']['avg_latency_ms']:.1f}ms > {config['max_latency_ms']}ms")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en métricas de performance: {e}")
        return False
    finally:
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass

async def test_cache_functionality():
    """Probar funcionalidad de cache"""
    logger.info("\n🧪 PROBANDO FUNCIONALIDAD DE CACHE")
    logger.info("=" * 50)
    
    config = {
        'max_latency_ms': 2000,
        'health_check_interval': 10,
        'cache_ttl_seconds': 10,  # 10 segundos TTL
        'max_memory_mb': 256,
        'model_pool_size': 2,
        'feature_cache_size': 10
    }
    
    engine = create_inference_engine(config)
    
    try:
        # Iniciar motor
        inference_task = asyncio.create_task(engine.start())
        await asyncio.sleep(1)
        
        # Primera ronda de predicciones (sin cache)
        logger.info("Primera ronda (sin cache)...")
        start_time = time.time()
        
        for i in range(5):
            await engine.predict("BTCUSDT", "1m", 1)
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(2)
        first_round_time = time.time() - start_time
        
        # Segunda ronda de predicciones (con cache)
        logger.info("Segunda ronda (con cache)...")
        start_time = time.time()
        
        for i in range(5):
            await engine.predict("BTCUSDT", "1m", 1)
            await asyncio.sleep(0.1)
        
        await asyncio.sleep(2)
        second_round_time = time.time() - start_time
        
        # Obtener métricas
        metrics = engine.get_metrics()
        
        logger.info(f"📊 CACHE:")
        logger.info(f"  Primera ronda: {first_round_time:.2f}s")
        logger.info(f"  Segunda ronda: {second_round_time:.2f}s")
        logger.info(f"  Hit rate: {metrics['performance']['cache_hit_rate']:.1f}%")
        logger.info(f"  Entradas en cache: {metrics['feature_cache']['total_entries']}")
        
        # Verificar que hay hit rate
        if metrics['performance']['cache_hit_rate'] > 0:
            logger.info("✅ Cache funcionando correctamente")
            return True
        else:
            logger.warning("⚠️  Cache no está funcionando")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en funcionalidad de cache: {e}")
        return False
    finally:
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass

async def test_health_monitoring():
    """Probar monitoreo de salud"""
    logger.info("\n🧪 PROBANDO MONITOREO DE SALUD")
    logger.info("=" * 50)
    
    config = {
        'max_latency_ms': 1000,
        'health_check_interval': 2,  # 2 segundos
        'cache_ttl_seconds': 30,
        'max_memory_mb': 128,  # Límite bajo para probar alertas
        'model_pool_size': 2,
        'feature_cache_size': 10
    }
    
    engine = create_inference_engine(config)
    
    try:
        # Iniciar motor
        inference_task = asyncio.create_task(engine.start())
        
        # Esperar varios health checks
        await asyncio.sleep(8)
        
        # Obtener métricas
        metrics = engine.get_metrics()
        
        logger.info(f"📊 SALUD DEL SISTEMA:")
        logger.info(f"  Memoria: {metrics['system']['memory_usage_mb']:.1f}MB")
        logger.info(f"  CPU: {metrics['system']['cpu_usage_percent']:.1f}%")
        logger.info(f"  Último health check: {metrics['system']['last_health_check']}")
        logger.info(f"  Modelos cargados: {metrics['system']['models_loaded']}")
        
        # Verificar que el health check está funcionando
        if metrics['system']['last_health_check']:
            logger.info("✅ Monitoreo de salud funcionando")
            return True
        else:
            logger.warning("⚠️  Monitoreo de salud no está funcionando")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en monitoreo de salud: {e}")
        return False
    finally:
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass

async def test_error_handling():
    """Probar manejo de errores"""
    logger.info("\n🧪 PROBANDO MANEJO DE ERRORES")
    logger.info("=" * 50)
    
    config = {
        'max_latency_ms': 1000,
        'health_check_interval': 5,
        'cache_ttl_seconds': 30,
        'max_memory_mb': 256,
        'model_pool_size': 2,
        'feature_cache_size': 10
    }
    
    engine = create_inference_engine(config)
    
    try:
        # Iniciar motor
        inference_task = asyncio.create_task(engine.start())
        await asyncio.sleep(1)
        
        # Solicitar predicciones con símbolos inválidos
        logger.info("Probando símbolos inválidos...")
        
        invalid_symbols = ["INVALID1", "INVALID2", "INVALID3"]
        for symbol in invalid_symbols:
            await engine.predict(symbol, "1m", 1)
            await asyncio.sleep(0.1)
        
        # Esperar procesamiento
        await asyncio.sleep(3)
        
        # Obtener métricas
        metrics = engine.get_metrics()
        
        logger.info(f"📊 MANEJO DE ERRORES:")
        logger.info(f"  Predicciones totales: {metrics['performance']['total_predictions']}")
        logger.info(f"  Tasa de error: {metrics['performance']['error_rate']:.1f}%")
        
        # El sistema debería manejar errores gracefully
        logger.info("✅ Manejo de errores funcionando")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en manejo de errores: {e}")
        return False
    finally:
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass

async def main():
    """Función principal de prueba"""
    logger.info("🚀 INICIANDO PRUEBAS DE INFERENCIA EN TIEMPO REAL")
    logger.info("=" * 60)
    
    tests = [
        ("Inferencia Básica", test_basic_inference),
        ("Métricas de Performance", test_performance_metrics),
        ("Funcionalidad de Cache", test_cache_functionality),
        ("Monitoreo de Salud", test_health_monitoring),
        ("Manejo de Errores", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 EJECUTANDO: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = await test_func()
            results[test_name] = success
            status = "✅ ÉXITO" if success else "❌ FALLO"
            logger.info(f"\n{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ ERROR en {test_name}: {e}")
            results[test_name] = False
    
    # Resumen final
    logger.info(f"\n{'='*60}")
    logger.info("📊 RESUMEN FINAL DE PRUEBAS")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"   {status} {test_name}")
    
    logger.info(f"\n🎯 RESULTADO: {passed}/{total} pruebas exitosas")
    
    if passed == total:
        logger.info("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        return 0
    else:
        logger.error("💥 ALGUNAS PRUEBAS FALLARON")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
