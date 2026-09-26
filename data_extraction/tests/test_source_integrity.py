import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import read_jsonl
from src.validators import check_compound_ids_in_registry, check_source_ids_exist

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _chunks():
    return list(read_jsonl(FIXTURES / "sample_chunks.jsonl"))


def test_all_cited_sources_exist():
    valid_ids = {c["source_id"] for c in _chunks()}
    records = [
        {
            "compound_id": "P0099_C1",
            "cation_source": ["P0099_Main_Page1_Paragraph1"],
            "connectivity_source": ["P0099_Main_Page2_Paragraph1"],
            "synthesis_source": ["P0099_Main_Page3_Paragraph1"],
        }
    ]
    assert check_source_ids_exist(records, valid_ids) == []


def test_detects_hallucinated_source_id():
    valid_ids = {c["source_id"] for c in _chunks()}
    records = [
        {
            "compound_id": "P0099_C1",
            "cation_source": ["P0099_Main_Page9_Paragraph99"],  # does not exist
            "connectivity_source": [],
            "synthesis_source": [],
        }
    ]
    violations = check_source_ids_exist(records, valid_ids)
    assert len(violations) == 1
    assert "P0099_Main_Page9_Paragraph99" in violations[0]


def test_compound_id_must_be_in_registry():
    records = [{"compound_id": "P0099_C1"}, {"compound_id": "P0099_C2"}]
    violations = check_compound_ids_in_registry(records, registry_compound_ids={"P0099_C1"})
    assert len(violations) == 1
    assert "P0099_C2" in violations[0]
