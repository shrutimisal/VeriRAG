from src.citations.claim_splitter import split_claims
from src.citations.verifier import UNVERIFIED, CitationVerifier, _parse_json_array
from tests.conftest import FakeLLM
from tests.test_citations import _chunks


def test_parses_fenced_json():
    assert _parse_json_array('```json\n[{"id": 1}]\n```') == [{"id": 1}]


def test_verdicts_mapped_and_labels_normalised():
    claims = split_claims("Claim number one here [1]. Claim number two here [2].", _chunks())
    llm = FakeLLM('[{"id":1,"label":"supported","explanation":"ok"},{"id":2,"label":"PARTIALLY_SUPPORTED","explanation":"half"}]')
    v = CitationVerifier(llm).verify(claims)
    assert [x.label for x in v] == ["Supported", "Partially Supported"]


def test_missing_and_garbled_become_unverified():
    claims = split_claims("Claim number one here [1]. Claim number two here [2].", _chunks())
    v = CitationVerifier(FakeLLM('[{"id":1,"label":"Contradicted","explanation":"no"},{"id":2,"label":"maybe"}]')).verify(claims)
    assert [x.label for x in v] == ["Contradicted", UNVERIFIED]
    v = CitationVerifier(FakeLLM("not json at all")).verify(claims)
    assert all(x.label == UNVERIFIED for x in v)
