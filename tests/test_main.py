import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from main import (
    get_segments,
    build_prompt,
    process_llm_responses,
    reconstruct_and_score,
    load_and_validate_dataset,
)


class TestGetSegments:
    def test_identical_strings_returns_equal_tag(self):
        segs = get_segments("hello world", "hello world", key=0)
        assert len(segs) == 1
        assert segs[0]["tag"] == "equal"

    def test_both_empty_returns_empty_list(self):
        assert get_segments("", "", key=0) == []

    def test_substitution_detected(self):
        segs = get_segments("hello world", "hello there", key=0)
        replace = [s for s in segs if s["tag"] == "replace"]
        assert len(replace) == 1
        assert replace[0]["reference"] == "world"
        assert replace[0]["prediction"] == "there"

    def test_insertion_detected(self):
        segs = get_segments("hello", "hello world", key=0)
        inserts = [s for s in segs if s["tag"] == "insert"]
        assert len(inserts) == 1
        assert inserts[0]["prediction"] == "world"

    def test_deletion_detected(self):
        segs = get_segments("hello world", "hello", key=0)
        deletes = [s for s in segs if s["tag"] == "delete"]
        assert len(deletes) == 1
        assert deletes[0]["reference"] == "world"

    def test_key_preserved_on_all_segments(self):
        segs = get_segments("a b", "a c", key="mykey")
        assert all(s["key"] == "mykey" for s in segs)

    def test_segment_indices_are_sequential(self):
        segs = get_segments("a b c", "a x c", key=0)
        assert [s["segment_idx"] for s in segs] == list(range(len(segs)))

    def test_position_metadata_is_correct(self):
        segs = get_segments("one two three", "one two three", key=0)
        eq = segs[0]
        assert eq["r_start"] == 0 and eq["r_end"] == 3
        assert eq["p_start"] == 0 and eq["p_end"] == 3


class TestBuildPrompt:
    def test_prompt_is_string(self):
        assert isinstance(build_prompt({"reference": "a", "prediction": "b"}), str)

    def test_reference_and_prediction_present(self):
        prompt = build_prompt({"reference": "doctor", "prediction": "डॉक्टर"})
        assert "doctor" in prompt
        assert "डॉक्टर" in prompt

    def test_index_is_zero(self):
        prompt = build_prompt({"reference": "a", "prediction": "b"})
        assert '"index": 0' in prompt

    def test_prompt_contains_template(self):
        # Prompt should start with the template content, not just the input JSON
        prompt = build_prompt({"reference": "a", "prediction": "b"})
        assert "INPUT" in prompt


class TestProcessLLMResponses:
    def _make_response(self, ref, pred, equivalent, reasoning="test"):
        return {
            "key": {"reference": ref, "prediction": pred},
            "response": {"equivalent": equivalent, "reasoning": reasoning},
        }

    def test_equivalent_segment_sets_flag(self):
        unique_segments = {("doctor", "डॉक्टर"): [{"row_idx": 0, "segment_idx": 1}]}
        responses = [self._make_response("doctor", "डॉक्टर", True)]
        flags, _ = process_llm_responses(responses, unique_segments)
        assert flags.get((0, 1)) is True

    def test_non_equivalent_segment_not_flagged(self):
        unique_segments = {("hello", "world"): [{"row_idx": 0, "segment_idx": 0}]}
        responses = [self._make_response("hello", "world", False)]
        flags, _ = process_llm_responses(responses, unique_segments)
        assert (0, 0) not in flags

    def test_one_segment_maps_to_multiple_rows(self):
        unique_segments = {
            ("doctor", "डॉक्टर"): [
                {"row_idx": 0, "segment_idx": 2},
                {"row_idx": 5, "segment_idx": 1},
            ]
        }
        responses = [self._make_response("doctor", "डॉक्टर", True)]
        flags, _ = process_llm_responses(responses, unique_segments)
        assert flags.get((0, 2)) is True
        assert flags.get((5, 1)) is True

    def test_log_records_contain_all_fields(self):
        unique_segments = {("a", "b"): [{"row_idx": 0, "segment_idx": 0}]}
        responses = [self._make_response("a", "b", False, reasoning="different words")]
        _, logs = process_llm_responses(responses, unique_segments)
        assert len(logs) == 1
        assert logs[0]["reference"] == "a"
        assert logs[0]["prediction"] == "b"
        assert logs[0]["equivalent"] is False
        assert logs[0]["reasoning"] == "different words"

    def test_malformed_response_skipped(self):
        unique_segments = {("a", "b"): [{"row_idx": 0, "segment_idx": 0}]}
        responses = [{"key": {}, "response": {}}]  # missing reference/prediction
        flags, logs = process_llm_responses(responses, unique_segments)
        assert len(flags) == 0
        assert len(logs) == 0

    def test_empty_inputs_return_empty(self):
        flags, logs = process_llm_responses([], {})
        assert flags == {}
        assert logs == []


class TestReconstructAndScore:
    def _df(self, ref, pred):
        return pd.DataFrame({"norm_reference": [ref], "norm_prediction": [pred]})

    def test_no_flags_preserves_original_prediction(self):
        df = self._df("hello world", "hello there")
        segs = get_segments("hello world", "hello there", key=0)
        result = reconstruct_and_score(df, {0: segs}, equivalent_flags={})
        assert result["corrected_prediction"].iloc[0] == "hello there"

    def test_equivalent_flag_replaces_prediction_with_reference(self):
        df = self._df("hello world", "hello there")
        segs = get_segments("hello world", "hello there", key=0)
        replace_idx = next(s["segment_idx"] for s in segs if s["tag"] == "replace")
        result = reconstruct_and_score(df, {0: segs}, equivalent_flags={(0, replace_idx): True})
        assert result["corrected_prediction"].iloc[0] == "hello world"
        assert result["corrected_wer"].iloc[0] == 0.0

    def test_identical_strings_give_zero_corrected_wer(self):
        df = self._df("hello world", "hello world")
        segs = get_segments("hello world", "hello world", key=0)
        result = reconstruct_and_score(df, {0: segs}, equivalent_flags={})
        assert result["corrected_wer"].iloc[0] == 0.0

    def test_corrected_wer_and_cer_columns_added(self):
        df = self._df("a b", "a c")
        segs = get_segments("a b", "a c", key=0)
        result = reconstruct_and_score(df, {0: segs}, equivalent_flags={})
        assert "corrected_wer" in result.columns
        assert "corrected_cer" in result.columns


class TestLoadAndValidateDataset:
    def test_missing_required_column_raises(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("transcription,prediction\nhello,hi\n")
        with pytest.raises(ValueError, match="Missing columns"):
            load_and_validate_dataset(str(csv), {"transcription", "prediction", "language"})

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_and_validate_dataset("/nonexistent/path.csv", {"col"})

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            load_and_validate_dataset(str(f), {"col"})

    def test_valid_csv_loads_correctly(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("ref,pred,audio,language\nhello,hi,a.wav,english\n")
        df = load_and_validate_dataset(str(csv), {"ref", "pred", "audio", "language"})
        assert len(df) == 1
        assert list(df.columns) == ["ref", "pred", "audio", "language"]

    def test_valid_jsonl_loads_correctly(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text('{"ref":"hello","pred":"hi","audio":"a.wav","language":"english"}\n')
        df = load_and_validate_dataset(str(jsonl), {"ref", "pred", "audio", "language"})
        assert len(df) == 1
