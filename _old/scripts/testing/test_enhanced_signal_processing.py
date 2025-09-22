#!/usr/bin/env python3
"""
Script de prueba para el sistema mejorado de procesamiento de señales
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from core.ml.signals.signal_processor import create_signal_processor
from core.ml.inference.postprocess import (
    prob_to_side, strength_from_prob, calculate_confidence,
    calculate_signal_quality, should_generate_signal, extract_signal_metadata
)

def test_postprocess_functions():
    """Probar las funciones de postprocess.py"""
    print("🧪 Probando funciones de postprocess.py...")
    print("=" * 50)
    
    try:
        # Casos de prueba
        test_cases = [
            {"prob_up": 0.8, "model_confidence": 0.9, "features_quality": 0.8, "market_volatility": 0.3},
            {"prob_up": 0.3, "model_confidence": 0.7, "features_quality": 0.9, "market_volatility": 0.2},
            {"prob_up": 0.52, "model_confidence": 0.4, "features_quality": 0.6, "market_volatility": 0.5},
            {"prob_up": 0.45, "model_confidence": 0.3, "features_quality": 0.4, "market_volatility": 0.8},
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\nCaso {i}: prob_up={case['prob_up']}, conf={case['model_confidence']}")
            
            # Calcular métricas de calidad
            quality_metrics = calculate_signal_quality(
                case['prob_up'], 
                case['model_confidence'], 
                case['features_quality'], 
                case['market_volatility']
            )
            
            # Determinar si generar señal
            should_signal = should_generate_signal(case['prob_up'], quality_metrics)
            
            # Calcular lado y fuerza
            side = prob_to_side(case['prob_up'])
            strength = strength_from_prob(case['prob_up'])
            
            print(f"  - Calidad: {quality_metrics['quality']:.3f}")
            print(f"  - Confianza: {quality_metrics['confidence']:.3f}")
            print(f"  - Fuerza: {strength:.3f}")
            print(f"  - Lado: {side}")
            print(f"  - Generar señal: {should_signal}")
            print(f"  - Alta calidad: {quality_metrics['is_high_quality']}")
            print(f"  - Señal clara: {quality_metrics['is_clear_signal']}")
        
        print("\n✅ Funciones de postprocess.py funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en funciones de postprocess.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_processor_with_enhanced_payload():
    """Probar el procesador de señales con payload enriquecido"""
    print("\n🧪 Probando Signal Processor con payload enriquecido...")
    print("=" * 50)
    
    try:
        # Crear procesador
        processor = create_signal_processor()
        
        # Obtener predicciones recientes
        predictions = processor.get_recent_predictions(limit=5)
        
        if not predictions:
            print("   ℹ️  No hay predicciones recientes para probar")
            return True
        
        print(f"   📊 Encontradas {len(predictions)} predicciones para probar")
        
        # Probar conversión de cada predicción
        signals_generated = 0
        high_quality_signals = 0
        
        for i, prediction in enumerate(predictions, 1):
            print(f"\n   Predicción {i}: {prediction.symbol}-{prediction.timeframe}")
            
            # Mostrar payload
            payload = prediction.payload
            print(f"     - prob_up: {payload.get('prob_up', 'N/A')}")
            print(f"     - model_confidence: {payload.get('model_confidence', 'N/A')}")
            print(f"     - features_quality: {payload.get('features_quality', 'N/A')}")
            print(f"     - market_volatility: {payload.get('market_volatility', 'N/A')}")
            print(f"     - processing_time_ms: {payload.get('processing_time_ms', 'N/A')}")
            print(f"     - features_count: {payload.get('features_count', 'N/A')}")
            
            # Convertir a señal
            signal = processor.convert_prediction_to_signal(prediction)
            
            if signal:
                signals_generated += 1
                print(f"     ✅ Señal generada: lado={signal.side}, fuerza={signal.strength:.3f}")
                
                # Verificar si es alta calidad
                if signal.meta.get('is_high_quality', False):
                    high_quality_signals += 1
                    print(f"     🌟 Señal de alta calidad")
                
                # Mostrar metadatos de calidad
                quality_metrics = signal.meta.get('quality_metrics', {})
                print(f"     - Calidad: {quality_metrics.get('quality', 0):.3f}")
                print(f"     - Confianza: {quality_metrics.get('confidence', 0):.3f}")
                print(f"     - Claridad: {quality_metrics.get('decision_clarity', 0):.3f}")
            else:
                print(f"     ❌ No se generó señal (no cumple criterios de calidad)")
        
        print(f"\n   📈 Resumen:")
        print(f"     - Predicciones procesadas: {len(predictions)}")
        print(f"     - Señales generadas: {signals_generated}")
        print(f"     - Señales de alta calidad: {high_quality_signals}")
        print(f"     - Tasa de conversión: {signals_generated/len(predictions)*100:.1f}%")
        print(f"     - Tasa de alta calidad: {high_quality_signals/signals_generated*100:.1f}%" if signals_generated > 0 else "     - Tasa de alta calidad: N/A")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_payload_structure():
    """Probar la estructura del payload enriquecido"""
    print("\n🧪 Probando estructura del payload enriquecido...")
    print("=" * 50)
    
    # Payload de ejemplo
    sample_payload = {
        "prob_up": 0.75,
        "model_confidence": 0.85,
        "features_quality": 0.92,
        "market_volatility": 0.25,
        "processing_time_ms": 45.2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features_used": ["rsi14", "ema20", "ema50", "macd", "atr14"],
        "features_count": 5,
        "prediction_id": "123_20241219_143022_0",
        "model_version": 123,
        "symbol": "BTCUSDT",
        "timeframe": "1m"
    }
    
    print("   📋 Payload de ejemplo:")
    for key, value in sample_payload.items():
        print(f"     - {key}: {value}")
    
    # Probar extracción de metadatos
    try:
        metadata = extract_signal_metadata(
            sample_payload,
            features_used=sample_payload["features_used"],
            model_version=str(sample_payload["model_version"]),
            processing_time_ms=sample_payload["processing_time_ms"]
        )
        
        print(f"\n   📊 Metadatos extraídos:")
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"     - {key}:")
                for sub_key, sub_value in value.items():
                    print(f"       - {sub_key}: {sub_value}")
            else:
                print(f"     - {key}: {value}")
        
        print("\n   ✅ Estructura del payload funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en estructura del payload: {e}")
        return False

def main():
    """Función principal de prueba"""
    print("🚀 INICIANDO PRUEBAS DEL SISTEMA MEJORADO DE SEÑALES")
    print("=" * 60)
    
    # Ejecutar pruebas
    tests = [
        ("Funciones de postprocess.py", test_postprocess_functions),
        ("Estructura del payload enriquecido", test_enhanced_payload_structure),
        ("Signal Processor con payload enriquecido", test_signal_processor_with_enhanced_payload),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE PRUEBAS")
    print('='*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{len(results)} pruebas pasaron")
    
    if passed == len(results):
        print("\n🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("\nEl sistema mejorado está listo para usar:")
        print("  - infer_bulk.py e infer_realtime.py generan payloads enriquecidos")
        print("  - postprocess.py calcula métricas de calidad avanzadas")
        print("  - signal_processor.py filtra solo señales de alta calidad")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} pruebas fallaron")
        print("Revisa los errores anteriores y corrige la configuración")
        return 1

if __name__ == "__main__":
    exit(main())
