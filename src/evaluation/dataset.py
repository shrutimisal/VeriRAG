"""Evaluation dataset loading/validation.

JSON format: a list of {"question": "...", "reference": "ground-truth answer"}.
('ground_truth' is accepted as an alias for 'reference'.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import get_settings
from src.utils.errors import EvaluationError


@dataclass
class EvalItem:
    question: str
    reference: str


def parse_dataset(raw: str | bytes) -> list[EvalItem]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Dataset is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("items") or data.get("data") or []
    if not isinstance(data, list) or not data:
        raise EvaluationError("Dataset must be a non-empty JSON list of {question, reference} objects.")
    items: list[EvalItem] = []
    for i, row in enumerate(data, 1):
        q = str(row.get("question", "")).strip() if isinstance(row, dict) else ""
        ref = str(row.get("reference", row.get("ground_truth", ""))).strip() if isinstance(row, dict) else ""
        if not q or not ref:
            raise EvaluationError(f"Dataset row {i} needs both 'question' and 'reference'.")
        items.append(EvalItem(q, ref))
    return items


def load_dataset(path: Path | str) -> list[EvalItem]:
    try:
        return parse_dataset(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationError(f"Cannot read dataset '{path}': {exc}") from exc


def sample_dataset_path() -> Path:
    return get_settings().eval_dir / "sample_eval_dataset.json"
