import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src"))

from utils.job_config import load_yaml, resolve_config, validate_config
from utils.job_store import ensure_dir, read_status, write_json, write_status


JOBS_DIR = ROOT_DIR / "data" / "jobs"
MAX_GRID_COMBOS = int(os.getenv("JOB_MAX_GRID_COMBOS", "100"))

app = FastAPI(title="Stock Model Training API", version="0.1.0")


class JobRequest(BaseModel):
    config_yaml: str | None = Field(default=None, description="Raw YAML config.")
    config: dict | None = Field(default=None, description="Config as JSON.")
    experiment_name: str | None = Field(default=None, description="Optional experiment block name.")
    tune_hyperparams: bool = Field(default=False)

    @model_validator(mode="after")
    def ensure_config_payload(self):
        if not self.config_yaml and not self.config:
            raise ValueError("Provide either config_yaml or config.")
        if self.config_yaml and self.config:
            raise ValueError("Provide only one of config_yaml or config.")
        return self


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _load_registries() -> tuple[set[str], set[str]]:
    from config.model_registry import MODEL_REGISTRY, PREPROCESS_REGISTRY

    return set(MODEL_REGISTRY.keys()), set(PREPROCESS_REGISTRY.keys())


def _write_submitted_config(job_dir: Path, payload: JobRequest) -> Path:
    config_path = job_dir / "config_submitted.yaml"
    if payload.config_yaml:
        config_path.write_text(payload.config_yaml)
    else:
        config_path.write_text(yaml.safe_dump(payload.config, sort_keys=False))
    return config_path


@app.get("/health")
def health_check():
    return {"status": "ok", "time": utc_now_iso()}


@app.post("/jobs")
def create_job(request: JobRequest):
    job_id = str(uuid4())
    job_dir = ensure_dir(JOBS_DIR / job_id)

    config_path = _write_submitted_config(job_dir, request)
    try:
        raw_config = load_yaml(config_path.read_text())
        resolved_config = resolve_config(raw_config, request.experiment_name)
        allowed_models, allowed_steps = _load_registries()
        validate_config(
            resolved_config,
            max_grid_combos=MAX_GRID_COMBOS,
            allowed_models=allowed_models,
            allowed_preprocess_steps=allowed_steps,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    write_json(
        job_dir / "request.json",
        {
            "job_id": job_id,
            "submitted_at": utc_now_iso(),
            "experiment_name": request.experiment_name,
            "tune_hyperparams": request.tune_hyperparams,
        },
    )
    write_status(job_dir, "queued", job_id=job_id, created_at=utc_now_iso())

    log_path = job_dir / "logs.txt"
    command = [
        sys.executable,
        str(ROOT_DIR / "src" / "training" / "job_runner.py"),
        "--job-id",
        job_id,
        "--config-path",
        str(config_path),
        "--output-dir",
        str(job_dir),
    ]
    if request.experiment_name:
        command.extend(["--experiment-name", request.experiment_name])
    if request.tune_hyperparams:
        command.append("--tune-hyperparams")

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

    write_status(
        job_dir,
        "running",
        job_id=job_id,
        started_at=utc_now_iso(),
        pid=process.pid,
        logs=str(log_path),
    )

    return {
        "job_id": job_id,
        "status": "running",
        "submitted_at": utc_now_iso(),
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    status = read_status(job_dir)
    response = {"job_id": job_id, "status": status.get("status", "unknown"), **status}

    results_path = job_dir / "results.json"
    if results_path.exists():
        response["result"] = json.loads(results_path.read_text())
    return response


@app.get("/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    artifacts = []
    for path in sorted(job_dir.glob("*")):
        if path.is_file():
            artifacts.append(str(path))
    return {"job_id": job_id, "artifacts": artifacts}
