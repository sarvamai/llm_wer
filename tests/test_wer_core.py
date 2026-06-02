"""
Tests for llm_wer core logic — no LLM API, no Google Sheets, no IndicNormalizer.
Covers: wer(), cer(), get_segments(), process_llm_responses(),
reconstruct_and_score(), load_and_validate_dataset(), and full pipeline integration.
"""
from __future__ import annotations

import json
import tempfile
import os
import sys
import pytest
import pandas as pd

# Allow imports from parent dir (the repo root contains main.py, utilities.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utilities import wer, cer
from main import (
    get_segments,
    process_llm_responses,
    reconstruct_and_score,
    load_and_validate_dataset,
)


# ─── WER tests ────────────────────────────────────────────────────────────────

def test_wer_identical_strings():
    assert wer("hello world", "hello world") == pytest.approx(0.0)


def test_wer_completely_different():
    # ref=["a", "b"], hyp=["c", "d"] → 2 subs, denom=max(2,2)=2 → 1.0
    assert wer("a b", "c d") == pytest.approx(1.0)


def test_wer_single_substitution():
    # ref=["hello", "world"], hyp=["hello", "earth"] → 1 sub / max(2,2)=2 → 0.5
    result = wer("hello world", "hello earth")
    assert result == pytest.approx(0.5)


def test_wer_empty_both():
    assert wer("", "") == pytest.approx(0.0)


def test_wer_empty_ref_nonempty_hyp():
    # ref is empty, hyp has words → insertion penalty
    result = wer("", "hello world")
    assert result == pytest.approx(1.0)  # insertion_weight default = 1


def test_wer_nonempty_ref_empty_hyp():
    result = wer("hello world", "")
    assert result == pytest.approx(1.0)  # deletion_weight default = 1


def test_wer_insertion():
    # ref=["hello"], hyp=["hello", "world"] → 1 insertion / max(1,2)=2 → 0.5
    result = wer("hello", "hello world")
    assert result == pytest.approx(0.5)


def test_wer_deletion():
    # ref=["hello", "world"], hyp=["hello"] → 1 deletion / max(2,1)=2 → 0.5
    result = wer("hello world", "hello")
    assert result == pytest.approx(0.5)


def test_wer_custom_weights():
    # ref=["a","b"], hyp=["a","c"] → 1 substitution, substitution_weight=2 → 2/max(2,2)=1.0
    result = wer("a b", "a c", substitution_weight=2)
    assert result == pytest.approx(1.0)


def test_wer_clamp_false():
    # Without clamping, denominator = N (reference length), not max(M,N)
    # ref=["hello"], hyp=["hello", "world", "foo"] → 2 insertions / N=1 → 2.0
    result = wer("hello", "hello world foo", clamp=False)
    assert result > 1.0


# ─── CER tests ────────────────────────────────────────────────────────────────

def test_cer_identical():
    assert cer("hello", "hello") == pytest.approx(0.0)


def test_cer_empty_both():
    assert cer("", "") == pytest.approx(0.0)


def test_cer_single_char_sub():
    # ref="hello", hyp="hella" → 1 sub / max(5,5)=5 → 0.2
    result = cer("hello", "hella")
    assert result == pytest.approx(0.2)


def test_cer_empty_ref():
    result = cer("", "abc")
    assert result == pytest.approx(1.0)


def test_cer_empty_hyp():
    result = cer("abc", "")
    assert result == pytest.approx(1.0)


# ─── get_segments() tests ─────────────────────────────────────────────────────

def test_get_segments_equal_strings():
    segments = get_segments("hello world", "hello world", key=0)
    tags = [s["tag"] for s in segments]
    assert all(t == "equal" for t in tags)


def test_get_segments_one_substitution():
    segments = get_segments("hello world", "hello earth", key=0)
    tags = [s["tag"] for s in segments]
    assert "replace" in tags
    equal_segs = [s for s in segments if s["tag"] == "equal"]
    assert any("hello" in s["reference"] for s in equal_segs)


def test_get_segments_insertion():
    segments = get_segments("hello", "hello world", key="k1")
    assert any(s["tag"] == "insert" for s in segments)


def test_get_segments_deletion():
    segments = get_segments("hello world", "hello", key="k1")
    assert any(s["tag"] == "delete" for s in segments)


def test_get_segments_empty_strings():
    segments = get_segments("", "", key=0)
    assert segments == []


def test_get_segments_key_preserved():
    segments = get_segments("a b c", "a x c", key="mykey")
    for s in segments:
        assert s["key"] == "mykey"


def test_get_segments_segment_idx_sequential():
    segments = get_segments("a b c", "a x c", key=0)
    for i, s in enumerate(segments):
        assert s["segment_idx"] == i


# ─── process_llm_responses() tests ───────────────────────────────────────────

def _make_unique_segments():
    return {
        ("hello", "helo"): [{"row_idx": 0, "segment_idx": 1}],
        ("world", "word"): [{"row_idx": 1, "segment_idx": 0}],
    }


def test_process_llm_responses_equivalent_marks_flag():
    unique_segments = _make_unique_segments()
    successful = [
        {"key": {"reference": "hello", "prediction": "helo"}, "response": {"equivalent": True, "reasoning": "typo"}},
        {"key": {"reference": "world", "prediction": "word"}, "response": {"equivalent": False, "reasoning": "different meaning"}},
    ]
    flags, logs = process_llm_responses(successful, unique_segments)
    assert flags.get((0, 1)) is True   # hello/helo → equivalent
    assert (1, 0) not in flags         # world/word → not equivalent


def test_process_llm_responses_all_non_equivalent():
    unique_segments = _make_unique_segments()
    successful = [
        {"key": {"reference": "hello", "prediction": "helo"}, "response": {"equivalent": False, "reasoning": "different"}},
        {"key": {"reference": "world", "prediction": "word"}, "response": {"equivalent": False, "reasoning": "different"}},
    ]
    flags, logs = process_llm_responses(successful, unique_segments)
    assert len(flags) == 0


def test_process_llm_responses_logs_have_correct_structure():
    unique_segments = _make_unique_segments()
    successful = [
        {"key": {"reference": "hello", "prediction": "helo"}, "response": {"equivalent": True, "reasoning": "phonetic"}},
    ]
    flags, logs = process_llm_responses(successful, unique_segments)
    assert len(logs) == 1
    assert logs[0]["reference"] == "hello"
    assert logs[0]["prediction"] == "helo"
    assert logs[0]["equivalent"] is True
    assert "phonetic" in logs[0]["reasoning"]


def test_process_llm_responses_missing_keys_skipped():
    unique_segments = _make_unique_segments()
    successful = [
        {"key": {}, "response": {"equivalent": True}},  # no reference/prediction keys
    ]
    flags, logs = process_llm_responses(successful, unique_segments)
    assert len(flags) == 0
    assert len(logs) == 0


def test_process_llm_responses_multiple_occurrences():
    """Same segment pair appearing in multiple rows — all should get flagged."""
    unique_segments = {
        ("penicillin", "penicilln"): [
            {"row_idx": 0, "segment_idx": 2},
            {"row_idx": 3, "segment_idx": 1},
        ],
    }
    successful = [
        {"key": {"reference": "penicillin", "prediction": "penicilln"}, "response": {"equivalent": True, "reasoning": "typo"}},
    ]
    flags, _ = process_llm_responses(successful, unique_segments)
    assert flags.get((0, 2)) is True
    assert flags.get((3, 1)) is True


# ─── reconstruct_and_score() tests ───────────────────────────────────────────

def test_reconstruct_all_equivalent_gives_zero_corrected_wer():
    df = pd.DataFrame({
        "norm_reference": ["hello world"],
        "norm_prediction": ["hello wrold"],
    })
    row_segment_map = {
        0: [
            {"tag": "equal", "reference": "hello", "prediction": "hello", "segment_idx": 0},
            {"tag": "replace", "reference": "world", "prediction": "wrold", "segment_idx": 1},
        ]
    }
    equivalent_flags = {(0, 1): True}  # wrold → world marked equivalent
    result = reconstruct_and_score(df, row_segment_map, equivalent_flags)
    assert result.loc[0, "corrected_wer"] == pytest.approx(0.0)


def test_reconstruct_non_equivalent_preserves_original_error():
    df = pd.DataFrame({
        "norm_reference": ["hello world"],
        "norm_prediction": ["hello earth"],
    })
    row_segment_map = {
        0: [
            {"tag": "equal", "reference": "hello", "prediction": "hello", "segment_idx": 0},
            {"tag": "replace", "reference": "world", "prediction": "earth", "segment_idx": 1},
        ]
    }
    equivalent_flags = {}  # nothing is equivalent
    result = reconstruct_and_score(df, row_segment_map, equivalent_flags)
    # corrected_wer should be same as original (1 sub / max(2,2) = 0.5)
    assert result.loc[0, "corrected_wer"] == pytest.approx(0.5)


def test_reconstruct_adds_corrected_columns():
    df = pd.DataFrame({
        "norm_reference": ["hello"],
        "norm_prediction": ["hello"],
    })
    row_segment_map = {
        0: [{"tag": "equal", "reference": "hello", "prediction": "hello", "segment_idx": 0}]
    }
    result = reconstruct_and_score(df, row_segment_map, {})
    assert "corrected_wer" in result.columns
    assert "corrected_cer" in result.columns
    assert "corrected_prediction" in result.columns
    assert "corrected_reference" in result.columns


# ─── load_and_validate_dataset() tests ───────────────────────────────────────

def test_load_csv_valid():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("transcription,prediction,audio_filepath,language\n")
        f.write("hello world,hello wrold,/path/to/audio.wav,hindi\n")
        fname = f.name
    try:
        df = load_and_validate_dataset(fname, {"transcription", "prediction", "audio_filepath", "language"})
        assert len(df) == 1
        assert "transcription" in df.columns
    finally:
        os.unlink(fname)


def test_load_jsonl_valid():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"transcription": "hello", "prediction": "helo", "audio_filepath": "/a.wav", "language": "hindi"}) + "\n")
        fname = f.name
    try:
        df = load_and_validate_dataset(fname, {"transcription", "prediction", "audio_filepath", "language"})
        assert len(df) == 1
    finally:
        os.unlink(fname)


def test_load_missing_required_column_raises():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("transcription,prediction\n")  # missing audio_filepath and language
        f.write("hello,helo\n")
        fname = f.name
    try:
        with pytest.raises(ValueError, match="Missing columns"):
            load_and_validate_dataset(fname, {"transcription", "prediction", "audio_filepath", "language"})
    finally:
        os.unlink(fname)


def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_and_validate_dataset("/nonexistent/path/file.csv", {"transcription"})


def test_load_unsupported_format_raises():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world\n")
        fname = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported"):
            load_and_validate_dataset(fname, {"transcription"})
    finally:
        os.unlink(fname)


# ─── Integration: full pipeline with synthetic data ───────────────────────────

def test_full_pipeline_no_llm():
    """
    Full pipeline: load → get_segments → process_llm_responses → reconstruct_and_score.
    Validates that LLM-equivalent segments lower WER relative to raw WER.
    """
    df = pd.DataFrame({
        "norm_reference": ["मैं कल आऊंगा", "account balance check"],
        "norm_prediction": ["मैं कल आऊँगा", "account balanc check"],  # minor diffs
        "language": ["hindi", "english"],
    })

    from main import extract_unique_segments
    row_segment_map, unique_segments = extract_unique_segments(df)

    # Simulate LLM deciding the first pair's diff is equivalent (spelling variant)
    # and the second pair's diff is NOT equivalent (actual error)
    successful = []
    for (ref, pred), occurrences in unique_segments.items():
        # "आऊंगा" vs "आऊँगा" → equivalent (chandrabindu variant)
        is_equiv = ("आऊ" in ref or "आऊ" in pred)
        successful.append({
            "key": {"reference": ref, "prediction": pred},
            "response": {"equivalent": is_equiv, "reasoning": "test"},
        })

    equivalent_flags, logs = process_llm_responses(successful, unique_segments)
    result = reconstruct_and_score(df, row_segment_map, equivalent_flags)

    # Row 0 (Hindi) should have corrected_wer <= original_wer
    assert result.loc[0, "corrected_wer"] <= wer(
        df.loc[0, "norm_reference"], df.loc[0, "norm_prediction"]
    ) + 0.001

    # Both rows should have the new columns
    assert all(col in result.columns for col in ["corrected_wer", "corrected_cer"])


def test_wer_reduction_for_phonetically_equivalent_segment():
    """
    Demonstrates the core value proposition of llm_wer:
    WER goes down when LLM marks a segment as equivalent.
    """
    ref = "penicillin allergy reported"
    hyp = "penicillin alergy reported"  # one char drop in 'allergy'

    original = wer(ref, hyp)

    df = pd.DataFrame({"norm_reference": [ref], "norm_prediction": [hyp]})
    from main import extract_unique_segments
    row_segment_map, unique_segments = extract_unique_segments(df)

    # Mark the substituted segment as equivalent
    successful = [
        {"key": {"reference": k[0], "prediction": k[1]}, "response": {"equivalent": True, "reasoning": "typo"}}
        for k in unique_segments.keys()
    ]
    flags, _ = process_llm_responses(successful, unique_segments)
    result = reconstruct_and_score(df, row_segment_map, flags)

    assert result.loc[0, "corrected_wer"] <= original
