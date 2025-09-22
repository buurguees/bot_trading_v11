# Esquema de dataset (por símbolo/TF de ejecución)

**Clave:** (symbol, timeframe, timestamp)

## Columnas base
- `close, open, high, low, volume` (de HistoricalData)
- Features del TF base: `rsi14, ema20, ema50, ema200, macd, macd_signal, macd_hist, atr14, bb_mid, bb_upper, bb_lower, obv, supertrend, st_dir` (de Features)

## Snapshots multi-TF (opcional pero recomendado)
Para cada TF alto (15m, 1h, 4h, 1d) añadimos las **últimas lecturas cerradas**:
- Sufijo: `_15m, _1h, _4h, _1d` (p. ej. `ema200_1h`, `st_dir_4h`).
- Método: `last_value` (forward-fill hasta nuevo cierre del TF alto).

## Label (direccional, ejemplo)
- Clasificación: `y = 1` si `close[t+1] > close[t]`, si no `0`.
- Alternativas: retorno a H barras, o triclase (sube/igual/baja) con umbral ε.

## Split temporal
- Walk-forward: entrenar en ventana antigua → validar en ventana reciente; repetir.
