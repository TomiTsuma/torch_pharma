from abc import ABC, abstractmethod

class Task(ABC):
    """Abstract base class for all tasks (prediction, generation, etc.)."""
    @abstractmethod
    def run(self, model, data_loader):
        pass
