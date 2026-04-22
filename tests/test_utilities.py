import pytest
from utilities import wer, cer


class TestWER:
    def test_identical_strings(self):
        assert wer("hello world", "hello world") == 0.0

    def test_both_empty(self):
        assert wer("", "") == 0.0

    def test_empty_reference_counts_as_insertion(self):
        assert wer("", "hello world") == 1.0

    def test_empty_hypothesis_counts_as_deletion(self):
        assert wer("hello world", "") == 1.0

    def test_one_substitution_out_of_two(self):
        # "world" → "there": 1 substitution, denom = max(2, 2) = 2
        assert wer("hello world", "hello there") == 0.5

    def test_all_substitutions(self):
        assert wer("one two three", "four five six") == 1.0

    def test_one_deletion_clamped(self):
        # "hello world" → "hello": 1 deletion, denom = max(1, 2) = 2
        assert wer("hello world", "hello") == 0.5

    def test_one_insertion_clamped(self):
        # "hello" → "hello world": 1 insertion, denom = max(2, 1) = 2
        assert wer("hello", "hello world") == 0.5

    def test_no_clamp_uses_reference_length(self):
        # deletion, denom = N = 2 (not clamped)
        assert wer("hello world", "hello", clamp=False) == 0.5

    def test_custom_substitution_weight(self):
        # 1 substitution, weight=2, denom = max(2,2) = 2  → 2/2 = 1.0
        assert wer("hello world", "hello there", substitution_weight=2) == 1.0

    def test_whitespace_stripped(self):
        assert wer("  hello  ", "  hello  ") == 0.0


class TestCER:
    def test_identical_strings(self):
        assert cer("hello", "hello") == 0.0

    def test_both_empty(self):
        assert cer("", "") == 0.0

    def test_empty_reference(self):
        assert cer("", "a") == 1.0

    def test_empty_hypothesis(self):
        assert cer("a", "") == 1.0

    def test_one_char_substitution(self):
        # "abc" → "axc": 1 sub, denom = max(3, 3) = 3
        assert abs(cer("abc", "axc") - 1 / 3) < 1e-9

    def test_all_chars_different(self):
        assert cer("abc", "xyz") == 1.0

    def test_one_char_deletion_clamped(self):
        # "ab" → "a": 1 deletion, denom = max(1, 2) = 2
        assert cer("ab", "a") == 0.5

    def test_custom_deletion_weight(self):
        # "ab" → "a": 1 deletion weight=2, denom = max(1,2)=2 → 2/2 = 1.0
        assert cer("ab", "a", deletion_weight=2) == 1.0
