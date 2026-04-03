# Installation

Torch Pharma supports Python 3.9+ and PyTorch 2.0+.

---

## Quick Install

The simplest way to install Torch Pharma is via pip:

```bash
pip install torch-pharma
```

---

## From Source

For the latest development version or to contribute:

```bash
git clone https://github.com/TomiTsuma/torch_pharma.git
cd torch_pharma
pip install -e .
```

---

## Requirements

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | >=2.0.0 | Deep learning framework |
| RDKit | latest | Cheminformatics |
| PyTorch Geometric | >=2.3.0 | Graph neural networks |
| NumPy | >=1.20 | Numerical computing |
| Pandas | >=1.3 | Data manipulation |
| PyYAML | >=5.0 | Configuration files |
| tqdm | latest | Progress bars |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| OpenBabel | Molecular file format conversion |
| wandb | Experiment tracking |
| pytest | Testing |
| black, isort, flake8 | Code formatting and linting |

---

## Platform-Specific Instructions

### Linux

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y openbabel

# Install Torch Pharma
pip install torch-pharma
```

### macOS

```bash
# Using Homebrew
brew install open-babel

# Install Torch Pharma
pip install torch-pharma
```

### Windows

Windows users should use WSL2 (Windows Subsystem for Linux) for the best experience, as some dependencies (like OpenBabel) are difficult to install natively on Windows.

---

## GPU Support

Torch Pharma automatically uses CUDA if available. To ensure GPU support:

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is not available, ensure you have the correct PyTorch version installed:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Verify Installation

After installation, verify that Torch Pharma is correctly installed:

```bash
python -c "import torch_pharma; print(torch_pharma.__version__)"
```

You should see the version number printed.

---

## Development Installation

For development, install with additional dependencies:

```bash
git clone https://github.com/TomiTsuma/torch_pharma.git
cd torch_pharma
pip install -e ".[dev]"
```

This installs:
- Testing dependencies (`pytest`, `pytest-cov`)
- Code quality tools (`black`, `isort`, `flake8`, `mypy`)
- Documentation tools (`mkdocs`, `mkdocs-material`, `mkdocstrings`)

---

## Troubleshooting

### RDKit Installation Issues

If you encounter issues with RDKit:

```bash
# Using conda (recommended)
conda install -c conda-forge rdkit

# Then install torch-pharma without dependencies
pip install torch-pharma --no-deps
```

### OpenBabel Installation Issues

If OpenBabel fails to install:

```bash
# Using conda
conda install -c conda-forge openbabel
```

### PyTorch Geometric Installation

For custom PyTorch Geometric installation:

```bash
# Install PyTorch first
pip install torch==2.0.0

# Install PyTorch Geometric
pip install torch-geometric

# Install extensions (optional but recommended)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Import Errors

If you see `ImportError` or `ModuleNotFoundError`:

1. Ensure you're using the correct Python environment
2. Try reinstalling: `pip uninstall torch-pharma && pip install torch-pharma`
3. Check that all dependencies are installed: `pip list | grep torch`

---

## Next Steps

Once installed, check out the [Quick Start Guide](getting_started/quickstart.md) to begin using Torch Pharma.
