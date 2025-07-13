#!/bin/bash

DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
(
    cd "$DIR"
    python -m ruff format
    python -m ruff check --fix
    python -m pyright
)
