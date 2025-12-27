FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    freetds-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

ENV UV_NO_DEV=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
