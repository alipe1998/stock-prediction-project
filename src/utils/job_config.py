import copy
from datetime import datetime
from itertools import product

import yaml


def load_yaml(text: str) -> dict:
    payload = yaml.safe_load(text)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Submitted YAML must parse to a dictionary.")
    return payload


def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_config(raw_config: dict, experiment_name: str | None) -> dict:
    if "default" in raw_config:
        base = raw_config.get("default") or {}
        if not isinstance(base, dict):
            raise ValueError("default config block must be a dictionary.")
        if experiment_name:
            if experiment_name not in raw_config:
                raise ValueError(f"Experiment '{experiment_name}' not found.")
            override = raw_config.get(experiment_name) or {}
            if not isinstance(override, dict):
                raise ValueError(f"Experiment '{experiment_name}' must be a dictionary.")
            return deep_merge(base, override)
        return copy.deepcopy(base)
    if experiment_name:
        raise ValueError("experiment_name provided but no default block exists.")
    return copy.deepcopy(raw_config)


def _parse_month(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"Invalid month format '{value}'. Expected YYYY-MM.") from exc


def _validate_param_grid(param_grid: dict, max_combos: int) -> None:
    if not isinstance(param_grid, dict):
        raise ValueError("model.params must be a dictionary.")
    total_combos = 1
    for name, values in param_grid.items():
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"model.params.{name} must be a non-empty list.")
        total_combos *= len(values)
        if total_combos > max_combos:
            raise ValueError(f"Parameter grid too large ({total_combos} combos).")


def validate_config(
    config: dict,
    max_grid_combos: int,
    allowed_models: set[str],
    allowed_preprocess_steps: set[str],
) -> None:
    if not isinstance(config, dict):
        raise ValueError("Resolved config must be a dictionary.")

    data_config = config.get("data")
    features_config = config.get("features")
    preprocessing_pipeline = config.get("preprocessing_pipeline")
    model_config = config.get("model")

    if not data_config or not isinstance(data_config, dict):
        raise ValueError("Config must include a 'data' dictionary.")
    if not features_config or not isinstance(features_config, dict):
        raise ValueError("Config must include a 'features' dictionary.")
    if preprocessing_pipeline is None or not isinstance(preprocessing_pipeline, list):
        raise ValueError("Config must include 'preprocessing_pipeline' as a list.")
    if not model_config or not isinstance(model_config, dict):
        raise ValueError("Config must include a 'model' dictionary.")

    overall_range = data_config.get("overall_date_range") or {}
    start_month = overall_range.get("start")
    end_month = overall_range.get("end")
    if not start_month or not end_month:
        raise ValueError("data.overall_date_range must include start and end.")
    _parse_month(start_month)
    _parse_month(end_month)

    rolling_window = data_config.get("rolling_window") or {}
    train_window = rolling_window.get("train_window_months")
    test_offset = rolling_window.get("test_offset")
    if not isinstance(train_window, int) or train_window <= 0:
        raise ValueError("data.rolling_window.train_window_months must be a positive int.")
    if not isinstance(test_offset, int) or test_offset <= 0:
        raise ValueError("data.rolling_window.test_offset must be a positive int.")

    base_features = features_config.get("base")
    if not isinstance(base_features, list) or not all(
        isinstance(item, str) for item in base_features
    ):
        raise ValueError("features.base must be a list of strings.")

    for step in preprocessing_pipeline:
        if not isinstance(step, str):
            raise ValueError("preprocessing_pipeline must contain only strings.")
        if step not in allowed_preprocess_steps:
            raise ValueError(f"Unknown preprocessing step: {step}")

    model_name = model_config.get("name")
    if not isinstance(model_name, str):
        raise ValueError("model.name must be a string.")
    if model_name not in allowed_models:
        raise ValueError(f"Unknown model name: {model_name}")

    param_grid = model_config.get("params", {})
    _validate_param_grid(param_grid, max_grid_combos)
