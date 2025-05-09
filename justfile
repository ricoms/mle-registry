# List all recipes
default:
    @just --list

deps-pre:
    uv python install 3.13
    uv python pin 3.13
    uv init

deps-install:
    uv sync

deps-install-dev:
    uv sync --dev

mlflow-up:
    cd mlflow-wine && \
        docker-compose up

mlflow-train:
    cd mlflow-wine && \
        uv run python train.py

mlflow-inference:
    cd mlflow-wine && \
        uv run python inference.py

dvc-up:
    cd dvc-wine \
    && docker-compose up

dvc-train:
    cd dvc-wine \
    && uv run python train.py \
    && uv run dvc add data/ \
    && uv run dvc add models/ \
    && uv run dvc add metrics/ \
    && uv run dvc add plots/ \
    && dvc push

dvc-inference:
    cd dvc-wine \
    && uv run python inference.py
