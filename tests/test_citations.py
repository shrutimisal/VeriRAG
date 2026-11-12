from src.citations.citation_builder import build_citations, is_not_found
from src.citations.claim_splitter import split_claims, split_sentences
from src.config import NOT_FOUND_MESSAGE
from src.models import RetrievedChunk
from src.utils.text import extract_refs, strip_markers


def _chunks(n=2):
    return [RetrievedChunk(i, f"id{i}", "d", f"doc{i}.pdf", i + 4, f"text {i}", 0.9, 0.1) for i in range(1, n + 1)]


def test_extract_refs_and_strip():
    assert extract_refs("A [1][2]. B [2, 3].") == [1, 2, 3]
    assert strip_markers("It is 28.4 BLEU [1].") == "It is 28.4 BLEU."


def test_citations_map_to_real_chunks_and_invented_refs_are_dropped():
    cites, invalid = build_citations("Fact one [1]. Fact two [2]. Made up [9].", _chunks())
    assert [(c.document, c.page, c.chunk_id) for c in cites] == [("doc1.pdf", 5, "id1"), ("doc2.pdf", 6, "id2")]
    assert invalid == [9]


def test_not_found_detection():
    assert is_not_found(NOT_FOUND_MESSAGE)
    assert not is_not_found("Paris [1].")


def test_sentence_split_keeps_decimals_and_markers():
    s = split_sentences("It scored 28.4 BLEU [1]. The model has 6 layers [2][3]. Done")
    assert s == ["It scored 28.4 BLEU [1].", "The model has 6 layers [2][3].", "Done"]


def test_claims_attach_passages_and_flag_uncited():
    claims = split_claims("The first claim is here [1]. The second claim has no citation.", _chunks())
    assert claims[0].refs == [1] and [p.ref for p in claims[0].passages] == [1] and not claims[0].uncited
    assert claims[1].uncited and len(claims[1].passages) == 2
