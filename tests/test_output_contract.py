"""Regression tests for the LLM prompt's structured output contract."""

import ast
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _llm_response_fields() -> set[str]:
    module = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    model = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMResponse"
    )
    return {
        field.target.id
        for field in model.body
        if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
    }


def test_prompt_output_example_matches_llm_response_schema() -> None:
    """Keep the instructions shown to the model aligned with validation."""
    prompt = (ROOT / "prompt_template.txt").read_text(encoding="utf-8")
    output_section = prompt.split("# Output Format", maxsplit=1)[1]
    example = output_section.rsplit("```json", maxsplit=1)[1].split("```", maxsplit=1)[0]

    example_fields = set(re.findall(r'^\s*"([^"\n]+)"\s*:', example, re.MULTILINE))

    assert example_fields == _llm_response_fields()
