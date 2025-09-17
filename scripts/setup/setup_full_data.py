#!/usr/bin/env python3
"""
setup_full_data.py - Script completo para configurar 365 días de datos históricos

Este script:
1. Descarga 365 días de datos históricos para todos los símbolos y timeframes
2. Calcula features (indicadores técnicos) para todos los datos
3. Verifica la cobertura de datos
4. Prepara el sistema para entrenamiento

Uso:
    python setup_full_data.py
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("setup_full_data")

def setup_logging():
    """Configurar logging para el script"""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Log a archivo
    file_handler = logging.FileHandler("logs/setup_full_data.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # Log a consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

async def download_historical_data():
    """Descargar datos históricos para todos los símbolos y timeframes"""
    logger.info("=== INICIANDO DESCARGA DE DATOS HISTÓRICOS ===")
    
    try:
        from core.data.historical_downloader import run_all
        await run_all(since_days=365)
        logger.info("✅ Descarga de datos históricos completada")
        return True
    except Exception as e:
        logger.error(f"❌ Error en descarga de datos históricos: {e}")
        return False

def calculate_features():
    """Calcular features para todos los símbolos y timeframes"""
    logger.info("=== INICIANDO CÁLCULO DE FEATURES ===")
    
    try:
        from core.features.indicator_calculator import main as calc_main
        calc_main()
        logger.info("✅ Cálculo de features completado")
        return True
    except Exception as e:
        logger.error(f"❌ Error en cálculo de features: {e}")
        return False

def verify_data_coverage():
    """Verificar cobertura de datos en la base de datos"""
    logger.info("=== VERIFICANDO COBERTURA DE DATOS ===")
    
    try:
        from core.data.database import get_engine
        from sqlalchemy import text
        
        engine = get_engine()
        
        with engine.begin() as conn:
            # Verificar datos históricos
            result = conn.execute(text("""
                SELECT symbol, timeframe, 
                       COUNT(*) as total_bars,
                       MIN(timestamp) as desde,
                       MAX(timestamp) as hasta
                FROM trading.historicaldata
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """))
            
            logger.info("📊 DATOS HISTÓRICOS (últimos 365 días):")
            for row in result:
                logger.info(f"  {row[0]} {row[1]}: {row[2]} barras | {row[3]} - {row[4]}")
            
            # Verificar features
            result = conn.execute(text("""
                SELECT symbol, timeframe, 
                       COUNT(*) as total_features,
                       MIN(timestamp) as desde,
                       MAX(timestamp) as hasta
                FROM trading.features
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """))
            
            logger.info("📈 FEATURES (últimos 365 días):")
            for row in result:
                logger.info(f"  {row[0]} {row[1]}: {row[2]} features | {row[3]} - {row[4]}")
            
            # Verificar cobertura por timeframe
            result = conn.execute(text("""
                SELECT timeframe, 
                       COUNT(DISTINCT symbol) as symbols,
                       ROUND(COUNT(*)::float / COUNT(DISTINCT symbol), 0) as avg_bars_per_symbol
                FROM trading.historicaldata
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY timeframe
                ORDER BY timeframe
            """))
            
            logger.info("📊 COBERTURA POR TIMEFRAME:")
            for row in result:
                logger.info(f"  {row[0]}: {row[1]} símbolos, {row[2]:.0f} barras promedio")
            
        return True
    except Exception as e:
        logger.error(f"❌ Error verificando cobertura: {e}")
        return False

def check_training_readiness():
    """Verificar que el sistema esté listo para entrenamiento"""
    logger.info("=== VERIFICANDO PREPARACIÓN PARA ENTRENAMIENTO ===")
    
    try:
        from core.data.database import get_engine
        from sqlalchemy import text
        
        engine = get_engine()
        
        with engine.begin() as conn:
            # Verificar que tenemos datos suficientes para entrenamiento
            result = conn.execute(text("""
                SELECT symbol, timeframe, COUNT(*) as bars
                FROM trading.features
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY symbol, timeframe
                HAVING COUNT(*) < 1000
                ORDER BY symbol, timeframe
            """))
            
            insufficient_data = list(result)
            if insufficient_data:
                logger.warning("⚠️  Símbolos/TFs con datos insuficientes (<1000 barras):")
                for row in insufficient_data:
                    logger.warning(f"  {row[0]} {row[1]}: {row[2]} barras")
            else:
                logger.info("✅ Todos los símbolos/TFs tienen datos suficientes para entrenamiento")
            
            # Verificar que tenemos al menos 6 meses de datos
            result = conn.execute(text("""
                SELECT symbol, timeframe, 
                       MIN(timestamp) as desde,
                       MAX(timestamp) as hasta,
                       EXTRACT(DAYS FROM (MAX(timestamp) - MIN(timestamp))) as dias
                FROM trading.features
                WHERE timestamp >= NOW() - INTERVAL '365 days'
                GROUP BY symbol, timeframe
                HAVING EXTRACT(DAYS FROM (MAX(timestamp) - MIN(timestamp))) < 180
                ORDER BY symbol, timeframe
            """))
            
            short_coverage = list(result)
            if short_coverage:
                logger.warning("⚠️  Símbolos/TFs con cobertura temporal insuficiente (<180 días):")
                for row in short_coverage:
                    logger.warning(f"  {row[0]} {row[1]}: {row[4]} días | {row[2]} - {row[3]}")
            else:
                logger.info("✅ Todos los símbolos/TFs tienen cobertura temporal suficiente")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error verificando preparación: {e}")
        return False

async def main():
    """Función principal"""
    setup_logging()
    
    logger.info("🚀 INICIANDO CONFIGURACIÓN COMPLETA DE DATOS")
    logger.info("=" * 60)
    
    # Paso 1: Descargar datos históricos
    success = await download_historical_data()
    if not success:
        logger.error("❌ Falló la descarga de datos históricos. Abortando.")
        return False
    
    # Paso 2: Calcular features
    success = calculate_features()
    if not success:
        logger.error("❌ Falló el cálculo de features. Abortando.")
        return False
    
    # Paso 3: Verificar cobertura
    success = verify_data_coverage()
    if not success:
        logger.error("❌ Falló la verificación de cobertura. Abortando.")
        return False
    
    # Paso 4: Verificar preparación para entrenamiento
    success = check_training_readiness()
    if not success:
        logger.error("❌ Falló la verificación de preparación. Abortando.")
        return False
    
    logger.info("=" * 60)
    logger.info("🎉 CONFIGURACIÓN COMPLETA EXITOSA")
    logger.info("✅ El sistema está listo para entrenamiento nocturno")
    logger.info("📊 365 días de datos históricos descargados")
    logger.info("📈 Features calculados para todos los timeframes")
    logger.info("🔧 Ejecuta 'python -m core.ml.training.daily_train.runner' para comenzar")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("⏹️  Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Error inesperado: {e}")
        sys.exit(1)
