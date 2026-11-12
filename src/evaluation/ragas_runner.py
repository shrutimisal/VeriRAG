"""RAGAS evaluation of the real VeriRAG pipeline.

Flow: for every (question, reference) run the REAL retrieval + generation, collect
(user_input, retrieved_contexts, response, reference), then let RAGAS compute the four metrics
with Gemini as the judge LLM and the local Sentence-Transformers model for embeddings.
Nothing here is hardcoded: scores only exist if RAGAS actually produced them.

RAGAS' API changes between releases. This module targets the 0.3.x style
(EvaluationDataset + metric classes + LangChain wrappers).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.config import Settings
from src.evaluation.dataset import EvalItem
from src.pipeline import VeriRAGPipeline
from src.utils.errors import EvaluationError, MissingAPIKeyError, VeriRAGError
from src.utils.logger import get_logger

log = get_logger(__name__)

# Our canonical metric keys -> substrings to look for in RAGAS' result column names.
_METRIC_COLUMN_HINTS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "relevancy",   # "answer_relevancy" or "response_relevancy" depending on version
    "context_precision": "precision",
    "context_recall": "recall",
}


@dataclass
class EvaluationReport:
    scores: dict[str, float | None]          # mean per metric (None if every row failed)
    per_sample: pd.DataFrame                 # question, response, reference, + metric columns
    errors: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"meta": self.meta, "scores": self.scores, "errors": self.errors,
             "per_sample": json.loads(self.per_sample.to_json(orient="records"))},
            indent=2,
        )

    def to_csv(self) -> str:
        return self.per_sample.to_csv(index=False)


def collect_samples(
    pipeline: VeriRAGPipeline, items: list[EvalItem], doc_ids: list[str] | None, top_k: int,
    progress=None,
) -> tuple[list[dict], list[str]]:
    """Run the real RAG pipeline on each question. Questions that fail are reported, not faked."""
    samples, errors = [], []
    for i, item in enumerate(items, 1):
        try:
            result = pipeline.ask(item.question, doc_ids=doc_ids, top_k=top_k)
            samples.append(
                {
                    "user_input": item.question,
                    "retrieved_contexts": [c.text for c in result.retrieved],
                    "response": result.answer,
                    "reference": item.reference,
                }
            )
        except MissingAPIKeyError:
            raise
        except VeriRAGError as exc:
            errors.append(f"Question {i} skipped (pipeline error): {exc}")
        if progress:
            progress(i / len(items), f"Generated answer {i}/{len(items)}")
    return samples, errors


def _build_ragas_components(settings: Settings, api_key: str):
    """Import lazily: RAGAS/LangChain are heavy and only needed for evaluation."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy,
        )
    except ImportError as exc:
        raise EvaluationError(
            f"RAGAS dependencies are missing or incompatible ({exc}). Run: pip install -r requirements.txt"
        ) from exc

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=api_key, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=settings.ragas_relevancy_strictness),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    return llm, embeddings, metrics


def run_evaluation(
    pipeline: VeriRAGPipeline, items: list[EvalItem], doc_ids: list[str] | None = None,
    top_k: int | None = None, progress=None,
) -> EvaluationReport:
    """Run the full evaluation. Raises EvaluationError/MissingAPIKeyError when it cannot complete."""
    if not pipeline.has_api_key:
        raise MissingAPIKeyError("A Gemini API key is required for answer generation and RAGAS judging.")
    if pipeline.store.count() == 0:
        raise EvaluationError("No documents are indexed. Upload the PDF(s) the dataset refers to first.")

    top_k = top_k or pipeline.settings.top_k
    samples, errors = collect_samples(pipeline, items, doc_ids, top_k, progress)
    if not samples:
        raise EvaluationError("No samples could be generated. " + " ".join(errors))

    if progress:
        progress(0.9, "Scoring with RAGAS (this can take a few minutes)...")
    llm, embeddings, metrics = _build_ragas_components(pipeline.settings, pipeline.llm.api_key)

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.run_config import RunConfig

        dataset = EvaluationDataset.from_list(samples)
        # Low parallelism + generous retries keeps us inside free-tier rate limits.
        result = evaluate(
            dataset, metrics=metrics, llm=llm, embeddings=embeddings,
            run_config=RunConfig(timeout=180, max_retries=8, max_workers=2),
            show_progress=False, raise_exceptions=False,
        )
        df = result.to_pandas()
    except Exception as exc:
        raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

    # Map RAGAS' column names onto our canonical metric keys.
    metric_cols: dict[str, str] = {}
    for key, hint in _METRIC_COLUMN_HINTS.items():
        for col in df.columns:
            if hint in col.lower() and col not in metric_cols.values():
                metric_cols[key] = col
                break
    scores: dict[str, float | None] = {}
    for key in _METRIC_COLUMN_HINTS:
        col = metric_cols.get(key)
        vals = pd.to_numeric(df[col], errors="coerce").dropna() if col else pd.Series(dtype=float)
        scores[key] = float(vals.mean()) if len(vals) else None
        if col and len(vals) < len(df):
            errors.append(f"{key}: {len(df) - len(vals)}/{len(df)} rows could not be scored (API errors/timeouts).")
        if not col:
            errors.append(f"{key}: metric column missing from RAGAS output.")
    if all(v is None for v in scores.values()):
        raise EvaluationError("RAGAS returned no valid scores. " + " ".join(errors))

    table = pd.DataFrame(
        {"question": [s["user_input"] for s in samples], "response": [s["response"] for s in samples],
         "reference": [s["reference"] for s in samples]}
    )
    for key, col in metric_cols.items():
        table[key] = pd.to_numeric(df[col], errors="coerce").values

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "questions": len(items), "evaluated": len(samples), "top_k": top_k,
        "llm": pipeline.settings.gemini_model, "embedding_model": pipeline.settings.embedding_model,
        "judge_note": "Gemini generates answers AND judges them: scores may be biased toward its own style.",
    }
    return EvaluationReport(scores={k: (None if v is None or math.isnan(v) else v) for k, v in scores.items()},
                            per_sample=table, errors=errors, meta=meta)


def save_report(report: EvaluationReport, directory) -> str:
    """Persist a report as JSON under data/evaluation/results/. Returns the file path."""
    out_dir = directory / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(report.to_json(), encoding="utf-8")
    return str(path)
