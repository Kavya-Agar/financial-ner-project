"""Regex-based postprocessing for financial NER model predictions.

Each `normalize_*` function takes a single entity value string and returns a
canonicalized version of it, or the original value unchanged if it doesn't
match a known pattern (never raises). `postprocess_entities` ties them
together: it merges adjacent/overlapping same-type entity spans and then
normalizes each merged span's text.

Entities are represented as plain dicts: `{"text": str, "label": str,
"start": int, "end": int}`. `start`/`end` are optional -- entities without
them are still normalized, just never merged with neighbors (there's no
positional information to merge on).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------

_SSN_RE = re.compile(r"^(\d{3})[-.\s]?(\d{2})[-.\s]?(\d{4})$")


def normalize_ssn(value: Any) -> Any:
    """Canonicalize a Social Security Number to XXX-XX-XXXX form.

    Accepts dashes, dots, spaces, or no separator at all between groups.
    Returns the input unchanged if it isn't a well-formed 3-2-4 digit SSN.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    match = _SSN_RE.match(value.strip())
    if not match:
        return value
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


# ---------------------------------------------------------------------------
# DATE
# ---------------------------------------------------------------------------

_DATE_FORMATS_WITH_YEAR = [
    "%B %d, %Y",
    "%B %d %Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y-%m-%d",
    "%m/%d/%y",
    "%m-%d-%y",
]


def normalize_date(value: Any) -> Any:
    """Canonicalize a fully-specified date (day + month + year) to ISO 8601.

    Dates without a year (e.g. "March 15") are ambiguous to canonicalize
    safely, so they -- like anything that doesn't parse -- are returned
    unchanged.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    text = value.strip()
    for fmt in _DATE_FORMATS_WITH_YEAR:
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return dt.strftime("%Y-%m-%d")
    return value


# ---------------------------------------------------------------------------
# AMOUNT
# ---------------------------------------------------------------------------

_PERCENT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*%$")
_AMOUNT_RE = re.compile(
    r"^(-?)\$?\s*([\d,]+(?:\.\d+)?)\s*(million|m|k|dollars)?$",
    re.IGNORECASE,
)


def normalize_amount(value: Any) -> Any:
    """Canonicalize a dollar amount or percentage string.

    Dollar amounts are rendered as "$X,XXX.XX" (with a leading "-" for
    negative amounts); percentages as "X.X%". Multi-decimal-point,
    empty, or otherwise malformed values are returned unchanged.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    text = value.strip()

    percent_match = _PERCENT_RE.match(text)
    if percent_match:
        try:
            num = float(percent_match.group(1))
        except ValueError:
            return value
        return f"{num:.1f}%"

    amount_match = _AMOUNT_RE.match(text)
    if not amount_match:
        return value
    sign, digits, suffix = amount_match.groups()
    digits_clean = digits.replace(",", "")
    try:
        num = float(digits_clean)
    except ValueError:
        return value

    if suffix:
        suffix_lower = suffix.lower()
        if suffix_lower in ("million", "m"):
            num *= 1_000_000
        elif suffix_lower == "k":
            num *= 1_000

    result = f"${num:,.2f}"
    if sign == "-":
        result = "-" + result
    return result


# ---------------------------------------------------------------------------
# ACCOUNT
# ---------------------------------------------------------------------------

_ACCOUNT_SEPARATOR_RE = re.compile(r"[\s-]")


def normalize_account(value: Any) -> Any:
    """Canonicalize an account/routing number to a plain digit string.

    Strips spaces and dashes; requires at least 6 remaining digits.
    Returns the input unchanged otherwise (e.g. contains letters, too
    short to plausibly be an account number).
    """
    if not isinstance(value, str) or not value.strip():
        return value
    stripped = _ACCOUNT_SEPARATOR_RE.sub("", value.strip())
    if stripped.isdigit() and len(stripped) >= 6:
        return stripped
    return value


# ---------------------------------------------------------------------------
# FORM
# ---------------------------------------------------------------------------

_SCHEDULE_RE = re.compile(r"^schedule\s+([A-Za-z0-9]+)$", re.IGNORECASE)
_FORM_RE = re.compile(r"^(?:form\s*)?([A-Za-z0-9-]+)$", re.IGNORECASE)

_FORM_CANONICAL = {
    "1040": "Form 1040",
    "1040ez": "Form 1040EZ",
    "w2": "W-2",
    "w-2": "W-2",
    "w4": "W-4",
    "w-4": "W-4",
    "1099": "Form 1099",
    "1099misc": "Form 1099-MISC",
    "1099-misc": "Form 1099-MISC",
    "1099int": "Form 1099-INT",
    "1099-int": "Form 1099-INT",
    "1098": "Form 1098",
    "941": "Form 941",
    "990": "Form 990",
    "8829": "Form 8829",
}


def normalize_form(value: Any) -> Any:
    """Canonicalize a tax/loan form identifier to a consistent display form.

    Recognizes common IRS form codes (with or without a "Form " prefix)
    and "Schedule X" references. Unrecognized values are left unchanged
    rather than guessed at.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    text = value.strip()

    schedule_match = _SCHEDULE_RE.match(text)
    if schedule_match:
        return f"Schedule {schedule_match.group(1).upper()}"

    form_match = _FORM_RE.match(text)
    if not form_match:
        return value
    code_key = form_match.group(1).lower()
    if code_key in _FORM_CANONICAL:
        return _FORM_CANONICAL[code_key]
    return value


# ---------------------------------------------------------------------------
# postprocess_entities: merge + normalize
# ---------------------------------------------------------------------------

_NORMALIZERS = {
    "SSN": normalize_ssn,
    "DATE": normalize_date,
    "AMOUNT": normalize_amount,
    "ACCOUNT": normalize_account,
    "FORM": normalize_form,
}


def _has_position(entity: dict[str, Any]) -> bool:
    return isinstance(entity.get("start"), (int, float)) and isinstance(entity.get("end"), (int, float))


_MAX_MERGE_GAP = 1  # tolerate a single separating character (e.g. one space)


def _merge_adjacent_same_type(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping/adjacent spans of the same entity label.

    "Adjacent" includes spans that touch or overlap, and spans separated
    by a single character (typically the whitespace between two words a
    model over-segmented into separate entities). Entities without
    start/end positions are passed through unmerged since there's nothing
    to compare positionally.
    """
    positioned = [e for e in entities if _has_position(e)]
    unpositioned = [e for e in entities if not _has_position(e)]

    positioned.sort(key=lambda e: (e["start"], e["end"]))

    merged: list[dict[str, Any]] = []
    for entity in positioned:
        if (
            merged
            and merged[-1].get("label") == entity.get("label")
            and entity["start"] - merged[-1]["end"] <= _MAX_MERGE_GAP
        ):
            prev = merged[-1]
            prev["end"] = max(prev["end"], entity["end"])
            prev_text = prev.get("text")
            new_text = entity.get("text")
            if isinstance(prev_text, str) and isinstance(new_text, str) and new_text not in prev_text:
                prev["text"] = f"{prev_text} {new_text}"
        else:
            merged.append(dict(entity))

    return merged + unpositioned


def postprocess_entities(entities: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Merge adjacent same-type spans and normalize each entity's text.

    Handles an empty/None entity list, entities with malformed values that
    don't match any known pattern (left as-is), and unicode text (no
    crash -- normalizers that don't recognize a pattern just pass it
    through).
    """
    if not entities:
        return []

    merged = _merge_adjacent_same_type(entities)

    result = []
    for entity in merged:
        label = entity.get("label")
        text = entity.get("text")
        normalizer = _NORMALIZERS.get(label)
        new_entity = dict(entity)
        if normalizer is not None and isinstance(text, str):
            try:
                new_entity["text"] = normalizer(text)
            except Exception:
                new_entity["text"] = text
        result.append(new_entity)
    return result
