import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.export_final_dataset import build_final_dataframe, build_final_records


def test_direct_final_export_flattens_verified_records():
    records = [
        {
            "paper_id": "P0001",
            "compound_id": "P0001_C1",
            "sb_halide_dimensionality_llm": "2D",
            "halides_present": ["Cl", "Br"],
        }
    ]

    result = build_final_dataframe(records)

    assert result.loc[0, "sb_halide_dimensionality"] == "2D"
    assert result.loc[0, "halides_present"] == "Cl; Br"


def test_direct_final_json_records_preserve_lists_and_use_output_names():
    records = [
        {
            "paper_id": "P0001",
            "compound_id": "P0001_C1",
            "sb_halide_dimensionality_llm": "2D",
            "halides_present": ["Cl", "Br"],
        }
    ]

    result = build_final_records(records)

    assert result[0]["sb_halide_dimensionality"] == "2D"
    assert result[0]["halides_present"] == ["Cl", "Br"]
    assert "sb_halide_dimensionality_llm" not in result[0]
