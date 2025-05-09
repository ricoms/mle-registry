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
    cd mlflow && \
        docker-compose up
