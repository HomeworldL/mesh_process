"""Common file IO utilities used across ingest and processing scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from yaml import Loader


def load_json(file_path: str | Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, file_path: str | Path) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_yaml(file_path: str | Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=Loader)
