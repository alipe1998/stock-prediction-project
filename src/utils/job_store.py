import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_status(job_dir: Path, status: str, **fields) -> None:
    status_path = job_dir / "status.json"
    payload = {}
    if status_path.exists():
        payload = read_json(status_path)
    payload.update(fields)
    payload["status"] = status
    payload["updated_at"] = utc_now_iso()
    write_json(status_path, payload)


def read_status(job_dir: Path) -> dict:
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return {}
    return read_json(status_path)
