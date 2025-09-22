def tf_to_seconds(tf: str) -> int:
    m = {"1m":60, "5m":300, "15m":900, "1h":3600, "4h":14400, "1d":86400}
    if tf not in m:
        raise ValueError(f"Timeframe no soportado: {tf}")
    return m[tf]
