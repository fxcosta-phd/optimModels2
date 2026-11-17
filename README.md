# optimModels
optimModels created by Sara Correia - https://pypi.org/project/optimModels/
Modified by Andre Fonseca to show over time concentrations and fluxes in simulation solution - https://github.com/andrepacofonseca/optimModels

Refactored by Felipe Costa to work with most recent Python version (Currently 3.14+).

## Installation

### Prerequisites
- **Python 3.12+**
- **Git**

### Install from GitHub
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install git+https://github.com/fxcosta-phd/optimModels2.git
```

### Local Development Installation
```bash
git clone https://github.com/fxcosta-phd/optimModels2.git
cd optimModels2

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[dev]"
pytest
```
