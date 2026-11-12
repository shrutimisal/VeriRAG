import pytest

from src.evaluation.dataset import load_dataset, parse_dataset, sample_dataset_path
from src.evaluation.ragas_runner import collect_samples, run_evaluation
from src.utils.errors import EvaluationError, MissingAPIKeyError


def test_sample_dataset_is_valid():
    items = load_dataset(sample_dataset_path())
    assert len(items) >= 5 and all(i.question and i.reference for i in items)


def test_dataset_validation():
    with pytest.raises(EvaluationError):
        parse_dataset("not json")
    with pytest.raises(EvaluationError):
        parse_dataset('[{"question": "only q"}]')
    assert parse_dataset('[{"question": "q", "ground_truth": "a"}]')[0].reference == "a"


def test_collect_samples_uses_real_pipeline(pipeline, pdf_bytes):
    pipeline.ingest("geo.pdf", pdf_bytes)
    samples, errors = collect_samples(pipeline, parse_dataset('[{"question":"capital of France?","reference":"Paris"}]'), None, 2)
    assert not errors and samples[0]["retrieved_contexts"] and samples[0]["response"]


def test_evaluation_refuses_without_documents_or_key(pipeline):
    items = parse_dataset('[{"question":"q","reference":"a"}]')
    with pytest.raises(EvaluationError):
        run_evaluation(pipeline, items)            # nothing indexed
    pipeline.llm.api_key = ""
    with pytest.raises(MissingAPIKeyError):
        run_evaluation(pipeline, items)
