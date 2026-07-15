"""Tests for src/postprocess/regex_rules.py, covering the required edge
cases: SSN with/without dashes, malformed SSN, all supported date formats
plus garbage, amounts with negative/percent/multiple decimals, empty
string, None-like missing fields, and unicode text."""

import pytest

from src.postprocess.regex_rules import (
    normalize_account,
    normalize_amount,
    normalize_date,
    normalize_form,
    normalize_ssn,
    postprocess_entities,
)

# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["123-45-6789", "123 45 6789", "123.45.6789", "123456789"],
)
def test_normalize_ssn_valid_variants(raw):
    assert normalize_ssn(raw) == "123-45-6789"


@pytest.mark.parametrize(
    "raw",
    ["123-45-678", "12-45-6789", "abc-de-fghi", "123--45-6789", ""],
)
def test_normalize_ssn_malformed_left_unchanged(raw):
    assert normalize_ssn(raw) == raw


def test_normalize_ssn_none_like_missing_field():
    assert normalize_ssn(None) is None


# ---------------------------------------------------------------------------
# DATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("March 15, 2024", "2024-03-15"),
        ("12/01/2023", "2023-12-01"),
        ("2024-03-15", "2024-03-15"),
        ("03-15-2024", "2024-03-15"),
        ("12/01/23", "2023-12-01"),
    ],
)
def test_normalize_date_all_supported_formats(raw, expected):
    assert normalize_date(raw) == expected


def test_normalize_date_garbage_left_unchanged():
    assert normalize_date("not a date at all") == "not a date at all"


def test_normalize_date_no_year_left_unchanged():
    # Ambiguous without a year -- don't fabricate one.
    assert normalize_date("March 15") == "March 15"


def test_normalize_date_empty_string():
    assert normalize_date("") == ""


def test_normalize_date_none_like_missing_field():
    assert normalize_date(None) is None


# ---------------------------------------------------------------------------
# AMOUNT
# ---------------------------------------------------------------------------


def test_normalize_amount_comma_thousands():
    assert normalize_amount("$5,000") == "$5,000.00"


def test_normalize_amount_already_has_cents():
    assert normalize_amount("$542.10") == "$542.10"


def test_normalize_amount_negative():
    assert normalize_amount("-$120.00") == "-$120.00"


def test_normalize_amount_percent():
    assert normalize_amount("15.5%") == "15.5%"


def test_normalize_amount_negative_percent():
    assert normalize_amount("-2.0%") == "-2.0%"


def test_normalize_amount_million_suffix():
    assert normalize_amount("$1.5 million") == "$1,500,000.00"


def test_normalize_amount_dollars_word():
    assert normalize_amount("2500 dollars") == "$2,500.00"


def test_normalize_amount_multiple_decimal_points_left_unchanged():
    raw = "$12.34.56"
    assert normalize_amount(raw) == raw


def test_normalize_amount_empty_string():
    assert normalize_amount("") == ""


def test_normalize_amount_none_like_missing_field():
    assert normalize_amount(None) is None


# ---------------------------------------------------------------------------
# ACCOUNT
# ---------------------------------------------------------------------------


def test_normalize_account_strips_dashes():
    assert normalize_account("1234-5678-9012") == "123456789012"


def test_normalize_account_too_short_left_unchanged():
    assert normalize_account("12345") == "12345"


def test_normalize_account_non_digits_left_unchanged():
    raw = "ABC123XYZ"
    assert normalize_account(raw) == raw


def test_normalize_account_none_like_missing_field():
    assert normalize_account(None) is None


# ---------------------------------------------------------------------------
# FORM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1040", "Form 1040"),
        ("Form 1040", "Form 1040"),
        ("w-2", "W-2"),
        ("W-2", "W-2"),
        ("1099-int", "Form 1099-INT"),
        ("Schedule c", "Schedule C"),
    ],
)
def test_normalize_form_known_codes(raw, expected):
    assert normalize_form(raw) == expected


def test_normalize_form_unknown_left_unchanged():
    assert normalize_form("Form XYZ-999") == "Form XYZ-999"


def test_normalize_form_none_like_missing_field():
    assert normalize_form(None) is None


# ---------------------------------------------------------------------------
# postprocess_entities: merging + normalization + robustness
# ---------------------------------------------------------------------------


def test_postprocess_entities_empty_list():
    assert postprocess_entities([]) == []
    assert postprocess_entities(None) == []


def test_postprocess_entities_merges_adjacent_same_type():
    entities = [
        {"text": "March", "label": "DATE", "start": 0, "end": 5},
        {"text": "15,", "label": "DATE", "start": 6, "end": 9},
        {"text": "2024", "label": "DATE", "start": 10, "end": 14},
    ]
    result = postprocess_entities(entities)
    assert len(result) == 1
    assert result[0]["start"] == 0
    assert result[0]["end"] == 14


def test_postprocess_entities_does_not_merge_different_types():
    entities = [
        {"text": "John", "label": "PER", "start": 0, "end": 4},
        {"text": "5", "label": "AMOUNT", "start": 5, "end": 6},
    ]
    result = postprocess_entities(entities)
    assert len(result) == 2


def test_postprocess_entities_merges_overlapping_spans():
    entities = [
        {"text": "123-45", "label": "SSN", "start": 0, "end": 6},
        {"text": "45-6789", "label": "SSN", "start": 4, "end": 11},
    ]
    result = postprocess_entities(entities)
    assert len(result) == 1
    assert result[0]["start"] == 0
    assert result[0]["end"] == 11


def test_postprocess_entities_normalizes_after_merge():
    entities = [{"text": "123-45-6789", "label": "SSN", "start": 0, "end": 11}]
    result = postprocess_entities(entities)
    assert result[0]["text"] == "123-45-6789"


def test_postprocess_entities_malformed_value_left_as_is_no_crash():
    entities = [{"text": "not-a-valid-ssn", "label": "SSN", "start": 0, "end": 15}]
    result = postprocess_entities(entities)
    assert result[0]["text"] == "not-a-valid-ssn"


def test_postprocess_entities_missing_start_end_no_crash():
    entities = [{"text": "1040", "label": "FORM"}]
    result = postprocess_entities(entities)
    assert result[0]["text"] == "Form 1040"


def test_postprocess_entities_unicode_text_no_crash():
    entities = [
        {"text": "José García", "label": "PER", "start": 0, "end": 11},
        {"text": "€5.000", "label": "AMOUNT", "start": 12, "end": 18},
    ]
    result = postprocess_entities(entities)
    assert result[0]["text"] == "José García"
    # Unrecognized currency symbol: left unchanged, no crash.
    assert result[1]["text"] == "€5.000"


def test_postprocess_entities_unknown_label_no_normalizer_passthrough():
    entities = [{"text": "New York", "label": "LOC", "start": 0, "end": 8}]
    result = postprocess_entities(entities)
    assert result[0]["text"] == "New York"
