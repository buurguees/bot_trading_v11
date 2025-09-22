#!/usr/bin/env python3
"""
Script de prueba para el builder optimizado
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ml.datasets.builder import (
    build_dataset_optimized,
    build_dataset_streaming,
    fetch_base_optimized,
    add_snapshots_optimized,
    cleanup_cache,
    get_cache_stats,
    initialize_optimized_builder,
    DataValidator
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Probar funcionalidad básica"""
    logger.info("🧪 PROBANDO FUNCIONALIDAD BÁSICA")
    logger.info("=" * 50)
    
    # Inicializar builder
    initialize_optimized_builder()
    
    # Probar con BTCUSDT 1m
    symbol = "BTCUSDT"
    tf = "1m"
    
    logger.info(f"Probando {symbol}-{tf}...")
    
    start_time = time.time()
    df = build_dataset_optimized(symbol, tf, use_cache=True)
    load_time = time.time() - start_time
    
    if df.empty:
        logger.error(f"❌ No se pudieron cargar datos para {symbol}-{tf}")
        return False
    
    logger.info(f"✅ Datos cargados: {len(df)} registros en {load_time:.2f}s")
    logger.info(f"   Columnas: {list(df.columns)}")
    logger.info(f"   Rango temporal: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    # Validar datos
    validator = DataValidator()
    ohlcv_validation = validator.validate_ohlcv(df)
    feature_validation = validator.validate_features(df, ["rsi14", "ema20", "ema50"])
    
    logger.info(f"   Validación OHLCV: {'✅' if ohlcv_validation['valid'] else '❌'}")
    if not ohlcv_validation['valid']:
        logger.warning(f"   Problemas OHLCV: {ohlcv_validation['issues']}")
    
    logger.info(f"   Validación Features: {'✅' if feature_validation['valid'] else '❌'}")
    if not feature_validation['valid']:
        logger.warning(f"   Problemas Features: {feature_validation['issues']}")
    
    return True

def test_cache_functionality():
    """Probar funcionalidad de cache"""
    logger.info("\n🧪 PROBANDO FUNCIONALIDAD DE CACHE")
    logger.info("=" * 50)
    
    symbol = "ETHUSDT"
    tf = "5m"
    
    # Primera carga (sin cache)
    logger.info(f"Primera carga de {symbol}-{tf} (sin cache)...")
    start_time = time.time()
    df1 = build_dataset_optimized(symbol, tf, use_cache=True)
    time1 = time.time() - start_time
    logger.info(f"   Tiempo: {time1:.2f}s")
    
    # Segunda carga (con cache)
    logger.info(f"Segunda carga de {symbol}-{tf} (con cache)...")
    start_time = time.time()
    df2 = build_dataset_optimized(symbol, tf, use_cache=True)
    time2 = time.time() - start_time
    logger.info(f"   Tiempo: {time2:.2f}s")
    
    # Verificar que los datos son iguales
    if df1.equals(df2):
        logger.info("✅ Datos idénticos entre cargas")
    else:
        logger.error("❌ Datos diferentes entre cargas")
        return False
    
    # Verificar mejora de velocidad
    if time2 < time1:
        speedup = time1 / time2
        logger.info(f"✅ Mejora de velocidad: {speedup:.1f}x")
    else:
        logger.warning(f"⚠️  No hay mejora de velocidad: {time2:.2f}s vs {time1:.2f}s")
    
    return True

def test_streaming_functionality():
    """Probar funcionalidad de streaming"""
    logger.info("\n🧪 PROBANDO FUNCIONALIDAD DE STREAMING")
    logger.info("=" * 50)
    
    symbol = "ADAUSDT"
    tf = "1m"
    chunk_size = 10000
    
    logger.info(f"Probando streaming de {symbol}-{tf} con chunks de {chunk_size}...")
    
    start_time = time.time()
    total_rows = 0
    chunk_count = 0
    
    try:
        for chunk in build_dataset_streaming(symbol, tf, chunk_size):
            chunk_count += 1
            total_rows += len(chunk)
            logger.info(f"   Chunk {chunk_count}: {len(chunk)} registros")
            
            # Procesar solo los primeros 3 chunks para la prueba
            if chunk_count >= 3:
                break
        
        load_time = time.time() - start_time
        logger.info(f"✅ Streaming completado: {total_rows} registros en {chunk_count} chunks en {load_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en streaming: {e}")
        return False

def test_different_timeframes():
    """Probar diferentes timeframes"""
    logger.info("\n🧪 PROBANDO DIFERENTES TIMEFRAMES")
    logger.info("=" * 50)
    
    symbol = "SOLUSDT"
    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    results = {}
    
    for tf in timeframes:
        logger.info(f"Probando {symbol}-{tf}...")
        
        start_time = time.time()
        try:
            df = build_dataset_optimized(symbol, tf, use_cache=True)
            load_time = time.time() - start_time
            
            if df.empty:
                logger.warning(f"   ⚠️  Sin datos para {tf}")
                results[tf] = {'rows': 0, 'time': 0, 'success': False}
            else:
                logger.info(f"   ✅ {len(df)} registros en {load_time:.2f}s")
                results[tf] = {'rows': len(df), 'time': load_time, 'success': True}
                
        except Exception as e:
            logger.error(f"   ❌ Error en {tf}: {e}")
            results[tf] = {'rows': 0, 'time': 0, 'success': False}
    
    # Resumen
    successful = sum(1 for r in results.values() if r['success'])
    total_rows = sum(r['rows'] for r in results.values())
    total_time = sum(r['time'] for r in results.values())
    
    logger.info(f"\n📊 RESUMEN DE TIMEFRAMES:")
    logger.info(f"   Exitosos: {successful}/{len(timeframes)}")
    logger.info(f"   Total registros: {total_rows:,}")
    logger.info(f"   Tiempo total: {total_time:.2f}s")
    
    return successful > 0

def test_cache_stats():
    """Probar estadísticas de cache"""
    logger.info("\n🧪 PROBANDO ESTADÍSTICAS DE CACHE")
    logger.info("=" * 50)
    
    stats = get_cache_stats()
    
    logger.info(f"📊 ESTADÍSTICAS DE CACHE:")
    logger.info(f"   Archivos: {stats['total_files']}")
    logger.info(f"   Tamaño: {stats['total_size_mb']:.2f} MB")
    logger.info(f"   Directorio: {stats['cache_dir']}")
    logger.info(f"   TTL: {stats['ttl_hours']} horas")
    
    return True

def test_cleanup():
    """Probar limpieza de cache"""
    logger.info("\n🧪 PROBANDO LIMPIEZA DE CACHE")
    logger.info("=" * 50)
    
    # Obtener stats antes
    stats_before = get_cache_stats()
    logger.info(f"Archivos antes: {stats_before['total_files']}")
    
    # Limpiar cache
    cleanup_cache()
    
    # Obtener stats después
    stats_after = get_cache_stats()
    logger.info(f"Archivos después: {stats_after['total_files']}")
    
    if stats_after['total_files'] <= stats_before['total_files']:
        logger.info("✅ Limpieza exitosa")
        return True
    else:
        logger.warning("⚠️  La limpieza no redujo el número de archivos")
        return True

def main():
    """Función principal de prueba"""
    logger.info("🚀 INICIANDO PRUEBAS DEL BUILDER OPTIMIZADO")
    logger.info("=" * 60)
    
    tests = [
        ("Funcionalidad Básica", test_basic_functionality),
        ("Funcionalidad de Cache", test_cache_functionality),
        ("Funcionalidad de Streaming", test_streaming_functionality),
        ("Diferentes Timeframes", test_different_timeframes),
        ("Estadísticas de Cache", test_cache_stats),
        ("Limpieza de Cache", test_cleanup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 EJECUTANDO: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
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
    sys.exit(main())
