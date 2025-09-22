### Comandos rápidos

Entrenamiento direccional:
```bash
python -m core.ml.training.train_direction
```

Entrenamiento nocturno (batch):
```bash
python -m core.ml.training.night_train.batch_train -c core/ml/training/night_train/night_plan.yaml --workers 2
```

Registrar versiones / registry helpers:
```python
from core.ml.training.registry import get_or_create_agent, register_version
```


