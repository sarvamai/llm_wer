"""Read the equivalence flag from an LLM response.

The response model field is `equivalent`. Older prompts asked for
`equivalence`, and cached rows may still use that key.
"""

from __future__ import annotations

from typing import Any


def verdict_is_equivalent(response: dict[str, Any]) -> bool:
    if "equivalent" in response:
        return bool(response["equivalent"])
    return bool(response.get("equivalence", False))
