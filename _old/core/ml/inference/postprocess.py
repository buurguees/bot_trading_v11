def prob_to_side(prob_up: float, buy_th: float = 0.55, sell_th: float = 0.45) -> int:
    """Convertir probabilidad a lado de la señal con umbrales configurables"""
    if prob_up >= buy_th:  return  +1
    if prob_up <= sell_th: return  -1
    return 0

def strength_from_prob(prob_up: float) -> float:
    """Calcular fuerza de la señal basada en la probabilidad"""
    return float(abs(prob_up - 0.5) * 2.0)  # 0..1

def calculate_confidence(prob_up: float, model_confidence: float = None) -> float:
    """Calcular confianza combinada de la predicción"""
    base_confidence = abs(prob_up - 0.5) * 2.0  # 0-1 basado en distancia del 0.5
    
    if model_confidence is not None:
        # Combinar confianza del modelo con la distancia de probabilidad
        return (base_confidence + model_confidence) / 2.0
    
    return base_confidence

def calculate_signal_quality(
    prob_up: float, 
    model_confidence: float = None,
    features_quality: float = None,
    market_volatility: float = None
) -> dict:
    """Calcular métricas de calidad de la señal"""
    
    # Confianza base
    confidence = calculate_confidence(prob_up, model_confidence)
    
    # Fuerza de la señal
    strength = strength_from_prob(prob_up)
    
    # Calidad de features (si está disponible)
    features_score = features_quality if features_quality is not None else 1.0
    
    # Ajuste por volatilidad del mercado (si está disponible)
    volatility_adjustment = 1.0
    if market_volatility is not None:
        # Reducir confianza en mercados muy volátiles
        volatility_adjustment = max(0.5, 1.0 - (market_volatility - 0.5))
    
    # Calidad final de la señal
    final_quality = confidence * features_score * volatility_adjustment
    
    # Score de decisión (qué tan clara es la decisión)
    decision_clarity = abs(prob_up - 0.5) * 2.0  # 0-1
    
    return {
        "confidence": min(confidence, 1.0),
        "strength": min(strength, 1.0),
        "quality": min(final_quality, 1.0),
        "decision_clarity": min(decision_clarity, 1.0),
        "features_score": features_score,
        "volatility_adjustment": volatility_adjustment,
        "is_high_quality": final_quality >= 0.7,
        "is_clear_signal": decision_clarity >= 0.6
    }

def should_generate_signal(
    prob_up: float,
    quality_metrics: dict,
    min_quality: float = 0.6,
    min_confidence: float = 0.5,
    min_strength: float = 0.3
) -> bool:
    """Determinar si se debe generar una señal basada en la calidad"""
    
    return (
        quality_metrics["quality"] >= min_quality and
        quality_metrics["confidence"] >= min_confidence and
        quality_metrics["strength"] >= min_strength and
        quality_metrics["is_clear_signal"]
    )

def extract_signal_metadata(
    prediction_payload: dict,
    features_used: list = None,
    model_version: str = None,
    processing_time_ms: float = None
) -> dict:
    """Extraer metadatos detallados para la señal"""
    
    prob_up = prediction_payload.get("prob_up", 0.5)
    model_confidence = prediction_payload.get("model_confidence")
    features_quality = prediction_payload.get("features_quality")
    market_volatility = prediction_payload.get("market_volatility")
    
    # Calcular métricas de calidad
    quality_metrics = calculate_signal_quality(
        prob_up, model_confidence, features_quality, market_volatility
    )
    
    # Determinar si generar señal
    should_signal = should_generate_signal(prob_up, quality_metrics)
    
    # Metadatos base
    metadata = {
        "prob_up": prob_up,
        "model_confidence": model_confidence,
        "features_quality": features_quality,
        "market_volatility": market_volatility,
        "quality_metrics": quality_metrics,
        "should_generate_signal": should_signal,
        "timestamp": prediction_payload.get("timestamp"),
        "processing_time_ms": processing_time_ms
    }
    
    # Añadir información del modelo si está disponible
    if model_version:
        metadata["model_version"] = model_version
    
    # Añadir features usadas si están disponibles
    if features_used:
        metadata["features_used"] = features_used
        metadata["features_count"] = len(features_used)
    
    # Añadir información de latencia si está disponible
    if processing_time_ms is not None:
        metadata["latency_ms"] = processing_time_ms
        metadata["is_fast_prediction"] = processing_time_ms < 1000  # < 1 segundo
    
    return metadata
