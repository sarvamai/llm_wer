import ast
from pathlib import Path

from verdict import verdict_is_equivalent

ROOT = Path(__file__).resolve().parents[1]


def test_schema_field_is_preferred_over_the_legacy_key():
    assert verdict_is_equivalent({"equivalent": True, "equivalence": False}) is True
    assert verdict_is_equivalent({"equivalent": False}) is False
    assert verdict_is_equivalent({"equivalence": True}) is True
    assert verdict_is_equivalent({"reasoning": "no flag"}) is False


def test_prompt_json_key_matches_the_response_model():
    prompt = (ROOT / "prompt_template.txt").read_text(encoding="utf-8")
    output = prompt.split("# Output Format", 1)[1]
    example = output.split("```json", 1)[1].split("```", 1)[0]
    assert '"equivalent"' in example
    assert "equivalence" not in output

    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    fields = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LLMResponse":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
    assert "equivalent" in fields
    assert "equivalence" not in fields
