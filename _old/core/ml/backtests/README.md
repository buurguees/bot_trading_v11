### Comandos rápidos

Backtests de planes (por tabla trading.TradePlans):

```bash
# Últimas 24h por defecto
python -m core.ml.backtests.backtest_plans --symbol BTCUSDT --tf 1m

# Rango concreto y parámetros realistas
python -m core.ml.backtests.backtest_plans \
  --symbol BTCUSDT --tf 1m \
  --from 2025-09-16 --to 2025-09-17 \
  --fees-bps 2 --slip-bps 2 --max-hold-bars 600 --funding-bps-8h 1.2
```

Backtests alternativos:

```bash
python -m core.ml.backtests.run_backtest
python -m core.ml.backtests.run_backtest_plans
```


