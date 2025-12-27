import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from data.fetch_data import get_db_engine, fetch_stock_data_with_features
from backtest.backtest import run_backtest
from utils.job_config import load_yaml, resolve_config, validate_config
from utils.job_store import write_json, write_status


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_summary(results_df: pd.DataFrame, start_month: str, end_month: str) -> dict:
    expected_months = len(pd.date_range(start=start_month, end=end_month, freq="MS"))
    if results_df.empty:
        return {
            "processed_months": 0,
            "expected_months": expected_months,
            "train_r2_mean": None,
            "test_r2_mean": None,
            "test_r2_best": None,
            "test_r2_worst": None,
        }

    summary = {
        "processed_months": int(results_df.shape[0]),
        "expected_months": expected_months,
        "train_r2_mean": float(results_df["train_r2"].mean()),
        "test_r2_mean": float(results_df["test_r2"].mean()),
        "test_r2_median": float(results_df["test_r2"].median()),
        "test_r2_std": float(results_df["test_r2"].std(ddof=0)),
        "test_r2_best": float(results_df["test_r2"].max()),
        "test_r2_worst": float(results_df["test_r2"].min()),
    }
    return summary


def write_predictions(results_df: pd.DataFrame, output_path: Path) -> int:
    count = 0
    with output_path.open("w") as handle:
        for _, row in results_df.iterrows():
            test_month = row.get("test_month")
            predictions = row.get("predictions") or []
            for pred in predictions:
                record = {
                    "test_month": test_month,
                    "ticker": pred.get("ticker"),
                    "actual_ret": pred.get("actual_ret"),
                    "predicted_ret": pred.get("predicted_ret"),
                }
                handle.write(json.dumps(record) + "\n")
                count += 1
    return count


def run_job(job_id: str, config_path: Path, output_dir: Path, experiment_name: str | None, tune_hyperparams: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    job_started_at = utc_now_iso()
    write_status(output_dir, "running", job_id=job_id, started_at=job_started_at, pid=os.getpid())

    raw_config = load_yaml(config_path.read_text())
    resolved_config = resolve_config(raw_config, experiment_name)

    from config.model_registry import MODEL_REGISTRY, PREPROCESS_REGISTRY

    validate_config(
        resolved_config,
        max_grid_combos=int(os.getenv("JOB_MAX_GRID_COMBOS", "100")),
        allowed_models=set(MODEL_REGISTRY.keys()),
        allowed_preprocess_steps=set(PREPROCESS_REGISTRY.keys()),
    )

    config_resolved_path = output_dir / "config_resolved.yaml"
    config_resolved_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False))

    load_dotenv(ROOT_DIR / ".env")
    server = os.getenv("DB_SERVER")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_NAME")
    if not all([server, username, password, database]):
        raise RuntimeError("Missing DB credentials in .env")

    data_config = resolved_config["data"]
    features = resolved_config["features"]["base"]
    preprocessing_pipeline = resolved_config["preprocessing_pipeline"]
    model_config = resolved_config["model"]

    engine = get_db_engine(server, username, password, database)
    df, _ = fetch_stock_data_with_features(engine, features, data_config)

    start_month = data_config["overall_date_range"]["start"]
    end_month = data_config["overall_date_range"]["end"]
    months_back = data_config["rolling_window"]["train_window_months"]

    start_time = time.perf_counter()
    results_df = run_backtest(
        df=df,
        model_name=model_config["name"],
        features=features,
        months_back=months_back,
        start_month=start_month,
        end_month=end_month,
        param_grid=model_config.get("params", {}),
        preprocessing_pipeline=preprocessing_pipeline,
        tune_hyperparams=tune_hyperparams,
    )
    elapsed = time.perf_counter() - start_time

    metrics_path = output_dir / "metrics.csv"
    metrics_df = results_df.drop(columns=["predictions"], errors="ignore").copy()
    if not metrics_df.empty:
        metrics_df["best_params"] = metrics_df["best_params"].apply(json.dumps)
    metrics_df.to_csv(metrics_path, index=False)

    predictions_path = output_dir / "predictions.jsonl"
    prediction_count = 0
    if results_df.empty:
        predictions_path.write_text("")
    else:
        prediction_count = write_predictions(results_df, predictions_path)

    summary = build_summary(results_df, start_month, end_month)
    timeseries = metrics_df.to_dict(orient="records") if not metrics_df.empty else []

    results_payload = {
        "job_id": job_id,
        "status": "finished",
        "started_at": job_started_at,
        "elapsed_sec": round(elapsed, 2),
        "summary": summary,
        "timeseries": timeseries,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "predictions_jsonl": str(predictions_path),
            "config_submitted": str(config_path),
            "config_resolved": str(config_resolved_path),
            "logs": str(output_dir / "logs.txt"),
        },
        "prediction_count": prediction_count,
    }

    write_json(output_dir / "results.json", results_payload)
    write_status(output_dir, "finished", finished_at=utc_now_iso(), elapsed_sec=round(elapsed, 2))


def main():
    parser = argparse.ArgumentParser(description="Run training/backtest job.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-name")
    parser.add_argument("--tune-hyperparams", action="store_true")
    args = parser.parse_args()

    try:
        run_job(
            job_id=args.job_id,
            config_path=Path(args.config_path),
            output_dir=Path(args.output_dir),
            experiment_name=args.experiment_name,
            tune_hyperparams=args.tune_hyperparams,
        )
    except Exception as exc:
        write_status(
            Path(args.output_dir),
            "failed",
            finished_at=utc_now_iso(),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
