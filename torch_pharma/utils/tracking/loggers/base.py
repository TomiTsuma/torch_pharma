from abc import ABC, abstractmethod
from typing import Optional
from torch_pharma.utils.tracking.store import ActivationStore

class ActivationLogger(ABC):
    """
    Abstract base class for all activation tracking loggers (MLflow, WandB, etc.).
    Follows the Strategy Pattern to decouple tracking logic from reporting logic.
    """
    
    @abstractmethod
    def log(self, store: ActivationStore, step: Optional[int] = None):
        """
        Logs the data present in the ActivationStore.
        
        Args:
            store: The populated ActivationStore containing layer statistics.
            step: Optional global step or epoch index representation.
        """
        pass
