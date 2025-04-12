# src/config/model_registry.py
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO)

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src"))

from models.mlp import MLPModel
from models.mlp_keras import MLPKerasModel
from data.preprocess import (
    drop_na, filter_price, quantile_transform,
    polynomial_features, encode_industry
)

MODEL_REGISTRY = {
    "mlp": MLPModel,
    "mlpk": MLPKerasModel
}

PREPROCESS_REGISTRY = {
    "drop_na": drop_na,
    "filter_price": filter_price,
    "quantile_transform": quantile_transform,
    "polynomial_features": polynomial_features,
    "encode_industry": encode_industry
}
