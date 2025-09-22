#!/usr/bin/env python3
"""
Script de prueba para el procesador de señales
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from core.ml.signals.signal_processor import create_signal_processor

def test_signal_processor():
    """Probar el procesador de señales"""
    print("🧪 Probando Signal Processor...")
    print("=" * 50)
    
    try:
        # Crear procesador
        print("1. Creando procesador...")
        processor = create_signal_processor()
        print("   ✅ Procesador creado correctamente")
        
        # Probar configuración
        print("2. Verificando configuración...")
        print(f"   - Batch size: {processor.config.batch_size}")
        print(f"   - Lookback minutes: {processor.config.lookback_minutes}")
        print(f"   - Min confidence: {processor.config.min_confidence}")
        print(f"   - Min strength: {processor.config.min_strength}")
        print(f"   - Filtros: {processor.config.filters}")
        print("   ✅ Configuración cargada correctamente")
        
        # Probar conexión a base de datos
        print("3. Probando conexión a base de datos...")
        try:
            with processor.engine.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                print("   ✅ Conexión a base de datos exitosa")
        except Exception as e:
            print(f"   ❌ Error de conexión: {e}")
            return False
        
        # Probar obtención de predicciones
        print("4. Probando obtención de predicciones...")
        predictions = processor.get_recent_predictions(limit=5)
        print(f"   - Predicciones encontradas: {len(predictions)}")
        if predictions:
            print(f"   - Última predicción: {predictions[0].timestamp}")
            print(f"   - Símbolo: {predictions[0].symbol}")
            print(f"   - Timeframe: {predictions[0].timeframe}")
        print("   ✅ Obtención de predicciones exitosa")
        
        # Probar procesamiento (si hay predicciones)
        if predictions:
            print("5. Probando procesamiento de señales...")
            stats = processor.process_predictions_to_signals()
            print(f"   - Predicciones procesadas: {stats.get('predictions', 0)}")
            print(f"   - Señales generadas: {stats.get('signals', 0)}")
            print(f"   - Señales guardadas: {stats.get('saved_signals', 0)}")
            print(f"   - Errores: {stats.get('errors', 0)}")
            print(f"   - Tiempo de procesamiento: {stats.get('processing_time', 0):.2f}s")
            print("   ✅ Procesamiento exitoso")
        else:
            print("5. No hay predicciones para procesar (esto es normal)")
            print("   ✅ Procesador listo para usar")
        
        print("\n" + "=" * 50)
        print("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN LAS PRUEBAS: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Probar la configuración"""
    print("\n🔧 Probando configuración...")
    
    try:
        processor = create_signal_processor()
        
        # Verificar configuración por defecto
        assert processor.config.batch_size > 0, "Batch size debe ser positivo"
        assert processor.config.lookback_minutes > 0, "Lookback minutes debe ser positivo"
        assert 0 <= processor.config.min_confidence <= 1, "Min confidence debe estar entre 0 y 1"
        assert 0 <= processor.config.min_strength <= 1, "Min strength debe estar entre 0 y 1"
        assert len(processor.config.filters) > 0, "Debe haber al menos un filtro"
        
        print("   ✅ Configuración válida")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en configuración: {e}")
        return False

def main():
    """Función principal de prueba"""
    print("🚀 INICIANDO PRUEBAS DEL PROCESADOR DE SEÑALES")
    print("=" * 60)
    
    # Ejecutar pruebas
    config_ok = test_configuration()
    processor_ok = test_signal_processor()
    
    print("\n" + "=" * 60)
    if config_ok and processor_ok:
        print("🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("\nEl procesador está listo para usar:")
        print("  python scripts/run_signal_processor.py --mode realtime")
        return 0
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON")
        print("\nRevisa los errores anteriores y corrige la configuración")
        return 1

if __name__ == "__main__":
    exit(main())
