from pathlib import Path

# Automatic home directory creation
TORCH_PHARMA_HOME = Path.home() / ".torch_pharma"
TORCH_PHARMA_HOME.mkdir(parents=True, exist_ok=True)

__version__ = "0.1.0"
