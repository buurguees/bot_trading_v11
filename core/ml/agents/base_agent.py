from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """Contrato mínimo para cualquier agente ML."""
    name: str = "BaseAgent"
    kind: str = "generic"  # direction|regime|volatility|execution|ensemble
    version: str = "v1.0.0"

    @abstractmethod
    def load(self, artifact_uri: Optional[str]) -> None:
        """Carga pesos/estado desde artifact_uri (o deja por defecto)."""
        ...

    @abstractmethod
    def predict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        row: dict con {symbol,timeframe,timestamp, ... features}
        return: payload JSON-serializable (p.ej., {'prob_up':0.61})
        """
        ...
