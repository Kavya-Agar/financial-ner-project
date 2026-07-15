"""Templated synthetic sentence generator for the 17-label EXTENDED_LABELS set.

This module produces bank-statement / tax-form / loan-application style
sentences with exact character-span gold entities, and provides the
tokenizer-alignment helpers needed to turn those spans into BIO token labels
for training (and, at inference time, to reconstruct entities from token
predictions).

Usage as a library:

    from src.data.synthetic_data import generate_synthetic_dataset, split_by_template
    examples = generate_synthetic_dataset(seed=42)
    train, val, test = split_by_template(examples)

Usage as a CLI:

    python -m src.data.synthetic_data --output-dir data/synthetic --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from src.data.label_schema import EXTENDED_LABEL2ID

# ---------------------------------------------------------------------------
# Raw vocabulary for filling templates
# ---------------------------------------------------------------------------

PERSON_NAMES = [
    "John Smith", "Sarah Johnson", "Michael Brown", "Lisa Davis", "David Wilson",
    "Emily Clark", "Robert Miller", "Jessica Garcia", "William Martinez", "Amanda Lee",
    "James Anderson", "Linda Thomas", "Christopher Taylor", "Karen White", "Daniel Harris",
    "Nancy Lewis", "Matthew Walker", "Susan Hall", "Anthony Young", "Betty King",
    "Jose Garcia", "Maria Rodriguez", "Wei Chen", "Priya Patel", "Ahmed Khan",
    "Fatima Ali", "Hiroshi Tanaka", "Yuki Sato", "Olga Ivanova", "Dmitri Petrov",
    "José García", "François Dubois", "Björn Nilsson", "Renée Müller",
    "Søren Andersen", "Émilie Rousseau", "Inés Fernández", "Nguyễn Văn An",
]

ORG_NAMES = [
    "Bank of America", "Wells Fargo", "JPMorgan Chase", "Citibank", "Goldman Sachs",
    "Microsoft Corporation", "Google LLC", "Amazon Inc", "Apple Inc", "Meta Platforms",
    "Charles Schwab", "Fidelity Investments", "Morgan Stanley", "US Bank", "PNC Bank",
    "TD Bank", "Capital One", "American Express", "Ally Financial", "Synchrony Bank",
]

LOCATION_NAMES = [
    "New York", "Los Angeles", "San Francisco", "Chicago", "Houston",
    "Seattle", "Boston", "Miami", "Denver", "Austin",
    "Portland", "Atlanta", "Phoenix", "Dallas", "Philadelphia",
    "San Diego", "Minneapolis", "Charlotte", "Detroit", "Sacramento",
]

FORM_CODES = [
    "Form 1040", "1040", "Form 1040EZ", "1040EZ",
    "Form W-2", "W-2", "Form W-4", "W-4",
    "Form 1099", "1099", "Form 1099-MISC", "1099-MISC", "Form 1099-INT", "1099-INT",
    "Form 8829", "8829", "Schedule C", "Schedule A",
    "Form 941", "941", "Form 990", "990", "Form 1098", "1098",
]

_DATE_FORMATTERS: list[Callable[[datetime], str]] = [
    lambda d: d.strftime("%B %d, %Y"),
    lambda d: d.strftime("%m/%d/%Y"),
    lambda d: d.strftime("%m-%d-%Y"),
    lambda d: d.strftime("%Y-%m-%d"),
    lambda d: d.strftime("%m/%d/%y"),
    lambda d: f"{d.strftime('%B')} {d.day}, {d.year}",
]

_DATE_START = datetime(2019, 1, 1)
_DATE_END = datetime(2025, 12, 31)


def _random_date(rng: random.Random) -> datetime:
    delta_days = (_DATE_END - _DATE_START).days
    return _DATE_START + timedelta(days=rng.randint(0, delta_days))


def generate_date(rng: random.Random) -> str:
    dt = _random_date(rng)
    formatter = rng.choice(_DATE_FORMATTERS)
    return formatter(dt)


def generate_person(rng: random.Random) -> str:
    return rng.choice(PERSON_NAMES)


def generate_organization(rng: random.Random) -> str:
    return rng.choice(ORG_NAMES)


def generate_location(rng: random.Random) -> str:
    return rng.choice(LOCATION_NAMES)


def generate_form(rng: random.Random) -> str:
    return rng.choice(FORM_CODES)


def generate_amount(rng: random.Random) -> str:
    kind = rng.choice(["comma", "cents", "comma_cents", "percent", "negative", "million", "dollars_word"])
    if kind == "comma":
        return f"${rng.randint(100, 999_999):,}"
    if kind == "cents":
        return f"${rng.randint(1, 999)}.{rng.randint(0, 99):02d}"
    if kind == "comma_cents":
        return f"${rng.randint(1_000, 999_999):,}.{rng.randint(0, 99):02d}"
    if kind == "percent":
        return f"{rng.uniform(0.1, 99.9):.1f}%"
    if kind == "negative":
        return f"-${rng.randint(1, 9_999)}.{rng.randint(0, 99):02d}"
    if kind == "million":
        return f"${rng.uniform(1.0, 9.9):.1f} million"
    return f"{rng.randint(100, 99_999)} dollars"


def generate_account(rng: random.Random) -> str:
    kind = rng.choice(["plain", "dashed", "routing"])
    if kind == "routing":
        return "".join(str(rng.randint(0, 9)) for _ in range(9))
    length = rng.randint(8, 12)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(length))
    if kind == "dashed":
        # Group into 4-digit chunks, e.g. 1234-5678-9012
        chunks = [digits[i : i + 4] for i in range(0, len(digits), 4)]
        return "-".join(chunks)
    return digits


def generate_ssn(rng: random.Random) -> str:
    part1 = "".join(str(rng.randint(0, 9)) for _ in range(3))
    part2 = "".join(str(rng.randint(0, 9)) for _ in range(2))
    part3 = "".join(str(rng.randint(0, 9)) for _ in range(4))
    sep = rng.choice(["-", " ", ".", ""])
    return f"{part1}{sep}{part2}{sep}{part3}"


# Maps a base placeholder field name (after stripping trailing digits, e.g.
# "date2" -> "date") to (generator function, entity label).
FIELD_GENERATORS: dict[str, tuple[Callable[[random.Random], str], str]] = {
    "person": (generate_person, "PER"),
    "organization": (generate_organization, "ORG"),
    "location": (generate_location, "LOC"),
    "amount": (generate_amount, "AMOUNT"),
    "date": (generate_date, "DATE"),
    "account": (generate_account, "ACCOUNT"),
    "ssn": (generate_ssn, "SSN"),
    "form": (generate_form, "FORM"),
}

_TRAILING_DIGITS_RE = re.compile(r"\d+$")


def _base_field_name(field_name: str) -> str:
    return _TRAILING_DIGITS_RE.sub("", field_name)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    template_id: str
    text: str
    doc_type: str


BANK_TEMPLATES = [
    Template("bank_deposit", "{person} deposited {amount} into account {account} on {date}.", "bank_statement"),
    Template("bank_withdrawal", "Account {account} processed a withdrawal of {amount} on {date}.", "bank_statement"),
    Template("bank_transfer", "{organization} transferred {amount} to {person} on {date}.", "bank_statement"),
    Template("bank_balance", "The account balance for {account} was {amount} as of {date}.", "bank_statement"),
    Template("bank_open", "{person} opened account {account} with {organization} on {date}.", "bank_statement"),
    Template("bank_wire", "{organization} processed a wire transfer of {amount} for {person} on {date}.", "bank_statement"),
    Template("bank_fee", "A service fee of {amount} was charged to account {account} on {date}.", "bank_statement"),
    Template("bank_interest", "Account {account} earned {amount} in interest during the statement period ending {date}.", "bank_statement"),
    Template("bank_relocate", "{person} of {location} closed account {account} with {organization} on {date}.", "bank_statement"),
    Template("bank_direct_deposit", "{organization} deposited {amount} directly into {person}'s account {account} on {date}.", "bank_statement"),
    Template("bank_overdraft", "Account {account} incurred an overdraft charge of {amount} on {date}.", "bank_statement"),
    Template("bank_statement_header", "Statement for account {account} held by {person} at {organization} covering {date}.", "bank_statement"),
    Template("bank_check", "{person} deposited a check for {amount} drawn on {organization} on {date}.", "bank_statement"),
    Template("bank_atm", "An ATM withdrawal of {amount} was made from account {account} in {location} on {date}.", "bank_statement"),
    Template("bank_routing", "Route funds using account {account} at {organization} for the {date} transfer of {amount}.", "bank_statement"),
    Template("bank_joint", "{person} and a co-owner share account {account} at {organization} in {location}.", "bank_statement"),
]

TAX_TEMPLATES = [
    Template("tax_filed_income", "{person} filed {form} reporting income of {amount} on {date}.", "tax_form"),
    Template("tax_ssn_required", "The taxpayer's Social Security Number {ssn} is required for {form}.", "tax_form"),
    Template("tax_owes", "{person} with SSN {ssn} owes {amount} according to {form}.", "tax_form"),
    Template("tax_deadline", "{form} must be filed by {date} for the tax year.", "tax_form"),
    Template("tax_income_reported", "Income of {amount} was reported on {form} by {person}.", "tax_form"),
    Template("tax_refund", "{person} received a refund of {amount} after filing {form} on {date}.", "tax_form"),
    Template("tax_employer", "{organization} issued {form} to {person} for income of {amount}.", "tax_form"),
    Template("tax_residence", "{person} of {location} filed {form} with SSN {ssn} on {date}.", "tax_form"),
    Template("tax_withholding", "Withholding of {amount} was reported for {person} on {form} filed {date}.", "tax_form"),
    Template("tax_extension", "{person} requested an extension for {form} beyond the {date} deadline.", "tax_form"),
    Template("tax_dependent", "{person} claimed a dependent credit of {amount} on {form} for tax year ending {date}.", "tax_form"),
    Template("tax_audit", "The IRS audited {form} filed by {person} with SSN {ssn} on {date}.", "tax_form"),
    Template("tax_amendment", "{person} amended {form} to correct income of {amount} on {date}.", "tax_form"),
    Template("tax_multiple_forms", "{person} submitted both {form} and a prior filing by {date}.", "tax_form"),
    Template("tax_state", "{person} of {location} owed {amount} in state tax reported on {form}.", "tax_form"),
    Template("tax_employer_ssn", "{organization} verified SSN {ssn} for {person} before issuing {form}.", "tax_form"),
]

LOAN_TEMPLATES = [
    Template("loan_apply", "{person} applied for a {amount} loan with {organization} on {date}.", "loan_application"),
    Template("loan_ssn_required", "The loan application {form} requires SSN {ssn} and income verification.", "loan_application"),
    Template("loan_approved", "{organization} approved {person} for {amount} on {date}.", "loan_application"),
    Template("loan_proceeds", "Account {account} will receive loan proceeds of {amount} on {date}.", "loan_application"),
    Template("loan_submit_form", "{person} submitted {form} to {organization} on {date}.", "loan_application"),
    Template("loan_denied", "{organization} denied the {amount} loan request from {person} on {date}.", "loan_application"),
    Template("loan_cosigner", "{person} and a cosigner from {location} applied for {amount} with {organization}.", "loan_application"),
    Template("loan_rate", "{organization} offered {person} a loan of {amount} at a rate disclosed on {date}.", "loan_application"),
    Template("loan_ssn_verify", "{organization} verified SSN {ssn} for {person} on {date} before approving {amount}.", "loan_application"),
    Template("loan_deposit_account", "Loan proceeds of {amount} were deposited into account {account} on {date}.", "loan_application"),
    Template("loan_relocation", "{person} relocating to {location} applied to {organization} for {amount} on {date}.", "loan_application"),
    Template("loan_form_income", "{person} attached {form} showing income of {amount} to the application dated {date}.", "loan_application"),
    Template("loan_default", "{organization} reported a default of {amount} on the account {account} as of {date}.", "loan_application"),
    Template("loan_prequalify", "{person} was pre-qualified by {organization} for up to {amount} on {date}.", "loan_application"),
    Template("loan_two_orgs", "{organization} referred {person} to a partner lender for {amount} on {date}.", "loan_application"),
    Template("loan_full", "{person} of {location}, SSN {ssn}, applied to {organization} for {amount} using {form} on {date}.", "loan_application"),
]

ALL_TEMPLATES: list[Template] = BANK_TEMPLATES + TAX_TEMPLATES + LOAN_TEMPLATES

_formatter = string.Formatter()


# ---------------------------------------------------------------------------
# Template instantiation with exact character spans
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    start: int
    end: int
    label: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "label": self.label, "text": self.text}


@dataclass
class Example:
    text: str
    entities: list[Entity]
    template_id: str
    doc_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "template_id": self.template_id,
            "doc_type": self.doc_type,
        }


def fill_template(template: Template, rng: random.Random) -> Example:
    """Instantiate a template, computing exact char-span gold entities.

    Spans are computed by literal string concatenation (never `str.find`),
    so there is no risk of off-by-one errors or accidental matches against
    an earlier occurrence of the same substring.
    """
    sentence_parts: list[str] = []
    entities: list[Entity] = []
    current_len = 0

    for literal_text, field_name, _format_spec, _conversion in _formatter.parse(template.text):
        sentence_parts.append(literal_text)
        current_len += len(literal_text)
        if field_name is None:
            continue
        base_name = _base_field_name(field_name)
        if base_name not in FIELD_GENERATORS:
            raise KeyError(f"Unknown placeholder '{field_name}' in template '{template.template_id}'")
        generator, label = FIELD_GENERATORS[base_name]
        value = generator(rng)
        start = current_len
        sentence_parts.append(value)
        current_len += len(value)
        end = current_len
        entities.append(Entity(start=start, end=end, label=label, text=value))

    text = "".join(sentence_parts)

    for ent in entities:
        actual = text[ent.start : ent.end]
        assert actual == ent.text, (
            f"Span mismatch in template '{template.template_id}': "
            f"expected {ent.text!r}, got {actual!r}"
        )

    return Example(text=text, entities=entities, template_id=template.template_id, doc_type=template.doc_type)


def generate_synthetic_dataset(
    seed: int = 42,
    min_instances_per_template: int = 10,
    max_instances_per_template: int = 14,
    templates: list[Template] | None = None,
) -> list[Example]:
    """Generate the full synthetic dataset (all templates, many instantiations each)."""
    rng = random.Random(seed)
    templates = templates if templates is not None else ALL_TEMPLATES
    examples: list[Example] = []
    for template in templates:
        n = rng.randint(min_instances_per_template, max_instances_per_template)
        for _ in range(n):
            examples.append(fill_template(template, rng))
    return examples


# ---------------------------------------------------------------------------
# Train / val / test split, grouped by template id (no template's
# instantiations may straddle a split boundary -- that would leak
# near-duplicate sentence structure across splits).
# ---------------------------------------------------------------------------


def split_by_template(
    examples: list[Example],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1234,
) -> tuple[list[Example], list[Example], list[Example]]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    template_ids = sorted({e.template_id for e in examples})
    rng = random.Random(seed)
    rng.shuffle(template_ids)

    n = len(template_ids)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    # Ensure every split gets at least one template if there are enough templates.
    n_train = min(n_train, n - 2) if n >= 3 else n_train
    n_val = min(n_val, max(n - n_train - 1, 0)) if n >= 3 else n_val

    train_ids = set(template_ids[:n_train])
    val_ids = set(template_ids[n_train : n_train + n_val])
    test_ids = set(template_ids[n_train + n_val :])

    train = [e for e in examples if e.template_id in train_ids]
    val = [e for e in examples if e.template_id in val_ids]
    test = [e for e in examples if e.template_id in test_ids]

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    return train, val, test


# ---------------------------------------------------------------------------
# Tokenizer alignment: char spans -> BIO token labels via word_ids()
# ---------------------------------------------------------------------------


def get_word_spans(text: str, tokenizer) -> tuple[list[str], list[tuple[int, int]]]:
    """Split text into "words" using the tokenizer's own pre-tokenizer.

    This mirrors exactly how the tokenizer will eventually chunk the text
    before WordPiece splitting, so punctuation-adjacent entities (e.g. a
    trailing period right after a form number) fall on separate words
    rather than being glued to the entity value.
    """
    pretokenized = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    words = [w for w, _ in pretokenized]
    spans = [(s, e) for _, (s, e) in pretokenized]
    return words, spans


def word_labels_from_entities(
    word_spans: list[tuple[int, int]], entities: list[dict[str, Any]]
) -> list[str]:
    """Assign one BIO label per word span, given char-span gold entities."""
    labels = ["O"] * len(word_spans)
    for ent in entities:
        ent_start, ent_end, ent_label = ent["start"], ent["end"], ent["label"]
        first = True
        for i, (w_start, w_end) in enumerate(word_spans):
            if w_end <= ent_start:
                continue
            if w_start >= ent_end:
                break
            labels[i] = f"{ent_label}_B" if first else f"{ent_label}_I"
            first = False
    return labels


def char_spans_to_token_labels(
    text: str,
    entities: list[dict[str, Any]],
    tokenizer,
    label2id: dict[str, int] | None = None,
    max_length: int | None = None,
):
    """Tokenize `text` and align char-span gold entities to BIO token labels.

    Returns the tokenizer `BatchEncoding` (from `is_split_into_words=True`)
    with an added "labels" key: one label id per token, -100 for special
    tokens and non-first subwords of a word (standard NER fine-tuning
    convention -- only the first subword of a word carries the loss).

    This function is dependency-injected on `tokenizer` so it can be unit
    tested with a tiny offline tokenizer fixture, and reused verbatim by
    the real DistilBERT tokenizer during training and by evaluation code
    that needs to align gold spans to model predictions.
    """
    label2id = label2id if label2id is not None else EXTENDED_LABEL2ID
    words, word_spans = get_word_spans(text, tokenizer)
    word_labels = word_labels_from_entities(word_spans, entities)

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    word_ids = encoding.word_ids()
    label_ids = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        elif word_id != previous_word_id:
            label_ids.append(label2id[word_labels[word_id]])
        else:
            label_ids.append(-100)
        previous_word_id = word_id

    encoding["labels"] = label_ids
    encoding["words"] = words
    encoding["word_spans"] = word_spans
    encoding["word_labels"] = word_labels
    return encoding


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _write_jsonl(examples: list[Example], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic financial-NER training data.")
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--min-instances-per-template", type=int, default=10)
    parser.add_argument("--max-instances-per-template", type=int, default=14)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    examples = generate_synthetic_dataset(
        seed=args.seed,
        min_instances_per_template=args.min_instances_per_template,
        max_instances_per_template=args.max_instances_per_template,
    )
    train, val, test = split_by_template(
        examples,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.split_seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train, output_dir / "train.jsonl")
    _write_jsonl(val, output_dir / "val.jsonl")
    _write_jsonl(test, output_dir / "test.jsonl")

    print(f"Generated {len(examples)} total examples across {len(ALL_TEMPLATES)} templates.")
    print(f"  train: {len(train)} examples -> {output_dir / 'train.jsonl'}")
    print(f"  val:   {len(val)} examples -> {output_dir / 'val.jsonl'}")
    print(f"  test:  {len(test)} examples -> {output_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
