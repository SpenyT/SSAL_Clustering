import os
from typing_extensions import Final

# rand config
SEED: Final[int] = 42

# data paths config
BASE_DIR:       Final[str] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR:       Final[str] = f"{BASE_DIR}/data"
CIFAR_DIR:      Final[str] = f"{DATA_DIR}/cifar-100-python"
VARIABLES_PATH: Final[str] = f"{DATA_DIR}/variables.pkl"