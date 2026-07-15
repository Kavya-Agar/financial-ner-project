"""Tests for src/data/synthetic_data.py: span generation, splitting, and
tokenizer alignment (using the tiny offline tokenizer fixture)."""

import pytest

from src.data.label_schema import EXTENDED_LABEL2ID
from src.data.synthetic_data import (
    ALL_TEMPLATES,
    Example,
    char_spans_to_token_labels,
    fill_template,
    generate_synthetic_dataset,
    get_word_spans,
    split_by_template,
    word_labels_from_entities,
)

# ---------------------------------------------------------------------------
# Span correctness
# ---------------------------------------------------------------------------


def test_fill_template_spans_match_text_slices():
    import random

    rng = random.Random(0)
    for template in ALL_TEMPLATES:
        for _ in range(5):
            example = fill_template(template, rng)
            for ent in example.entities:
                assert example.text[ent.start : ent.end] == ent.text
                assert ent.label in {
                    "PER", "ORG", "LOC", "AMOUNT", "DATE", "ACCOUNT", "SSN", "FORM",
                }


def test_fill_template_no_gaps_or_overlaps_within_template():
    import random

    rng = random.Random(1)
    example = fill_template(ALL_TEMPLATES[0], rng)
    entities = sorted(example.entities, key=lambda e: e.start)
    for a, b in zip(entities, entities[1:]):
        assert a.end <= b.start


def test_generate_synthetic_dataset_minimum_size():
    examples = generate_synthetic_dataset(seed=42, min_instances_per_template=10, max_instances_per_template=14)
    assert len(examples) >= 400
    for ex in examples:
        for ent in ex.entities:
            assert ex.text[ent.start : ent.end] == ent.text


def test_generate_synthetic_dataset_covers_all_doc_types():
    examples = generate_synthetic_dataset(seed=7)
    doc_types = {ex.doc_type for ex in examples}
    assert doc_types == {"bank_statement", "tax_form", "loan_application"}


def test_unknown_placeholder_raises():
    from src.data.synthetic_data import Template

    bad_template = Template("bad", "{nonsense} did something.", "bank_statement")
    import random

    with pytest.raises(KeyError):
        fill_template(bad_template, random.Random(0))


# ---------------------------------------------------------------------------
# Split: no template instantiation leaks across splits
# ---------------------------------------------------------------------------


def test_split_by_template_no_leakage():
    examples = generate_synthetic_dataset(seed=42)
    train, val, test = split_by_template(examples)

    train_ids = {e.template_id for e in train}
    val_ids = {e.template_id for e in val}
    test_ids = {e.template_id for e in test}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    # every example must be accounted for exactly once
    assert len(train) + len(val) + len(test) == len(examples)


def test_split_by_template_roughly_matches_fractions():
    examples = generate_synthetic_dataset(seed=42)
    train, val, test = split_by_template(examples, train_frac=0.7, val_frac=0.15, test_frac=0.15)
    total = len(examples)
    assert 0.55 < len(train) / total < 0.85
    assert 0.05 < len(val) / total < 0.30
    assert 0.05 < len(test) / total < 0.30


def test_split_by_template_bad_fractions_raises():
    examples = generate_synthetic_dataset(seed=1)
    with pytest.raises(ValueError):
        split_by_template(examples, train_frac=0.5, val_frac=0.3, test_frac=0.3)


def test_split_by_template_is_deterministic_given_seed():
    examples = generate_synthetic_dataset(seed=42)
    train1, val1, test1 = split_by_template(examples, seed=99)
    train2, val2, test2 = split_by_template(examples, seed=99)
    assert [e.text for e in train1] == [e.text for e in train2]
    assert [e.text for e in val1] == [e.text for e in val2]
    assert [e.text for e in test1] == [e.text for e in test2]


# ---------------------------------------------------------------------------
# Word-level BIO labeling from char spans (no tokenizer needed)
# ---------------------------------------------------------------------------


def test_word_labels_from_entities_basic():
    text = "John Smith works there"
    # word spans for "John", "Smith", "works", "there"
    word_spans = [(0, 4), (5, 10), (11, 16), (17, 22)]
    entities = [{"start": 0, "end": 10, "label": "PER"}]
    labels = word_labels_from_entities(word_spans, entities)
    assert labels == ["PER_B", "PER_I", "O", "O"]


def test_word_labels_entity_at_string_start():
    word_spans = [(0, 4), (5, 7)]
    entities = [{"start": 0, "end": 4, "label": "FORM"}]
    labels = word_labels_from_entities(word_spans, entities)
    assert labels[0] == "FORM_B"


def test_word_labels_entity_at_string_end():
    word_spans = [(0, 5), (6, 10)]
    entities = [{"start": 6, "end": 10, "label": "PER"}]
    labels = word_labels_from_entities(word_spans, entities)
    assert labels == ["O", "PER_B"]


# ---------------------------------------------------------------------------
# Tokenizer alignment via word_ids() -- uses the tiny offline tokenizer
# ---------------------------------------------------------------------------


def test_get_word_spans_splits_punctuation(tiny_tokenizer):
    text = "form 1040."
    words, spans = get_word_spans(text, tiny_tokenizer)
    assert words == ["form", "1040", "."]
    for word, (start, end) in zip(words, spans):
        assert text[start:end] == word


def test_char_spans_to_token_labels_punctuation_adjacent_entity(tiny_tokenizer):
    text = "form 1040."
    entities = [{"start": 5, "end": 9, "label": "FORM", "text": "1040"}]
    encoding = char_spans_to_token_labels(text, entities, tiny_tokenizer, label2id=EXTENDED_LABEL2ID)
    assert encoding["word_labels"] == ["O", "FORM_B", "O"]
    # the trailing period must not be swept into the entity
    assert "FORM_I" not in encoding["word_labels"]


def test_char_spans_to_token_labels_multi_token_word(tiny_tokenizer):
    text = "wash reported by jose garcia"
    # "wash" alone is a distinct vocab word here, standing in for an entity
    entities = [{"start": 0, "end": 4, "label": "ORG", "text": "wash"}]
    encoding = char_spans_to_token_labels(text, entities, tiny_tokenizer, label2id=EXTENDED_LABEL2ID)
    tokens = encoding.tokens()
    labels = encoding["labels"]
    assert tokens[0] == "[CLS]"
    # first real token should carry ORG_B
    assert labels[1] == EXTENDED_LABEL2ID["ORG_B"]


def test_char_spans_to_token_labels_entity_at_start_and_end(tiny_tokenizer):
    text = "1040 was filed by jose"
    entities = [
        {"start": 0, "end": 4, "label": "FORM", "text": "1040"},
        {"start": 19, "end": 23, "label": "PER", "text": "jose"},
    ]
    encoding = char_spans_to_token_labels(text, entities, tiny_tokenizer, label2id=EXTENDED_LABEL2ID)
    word_labels = encoding["word_labels"]
    assert word_labels[0] == "FORM_B"
    assert word_labels[-1] == "PER_B"


def test_char_spans_to_token_labels_special_tokens_get_ignore_index(tiny_tokenizer):
    text = "john smith"
    entities = [{"start": 0, "end": 10, "label": "PER", "text": "john smith"}]
    encoding = char_spans_to_token_labels(text, entities, tiny_tokenizer, label2id=EXTENDED_LABEL2ID)
    labels = encoding["labels"]
    tokens = encoding.tokens()
    assert tokens[0] == "[CLS]"
    assert labels[0] == -100
    assert tokens[-1] == "[SEP]"
    assert labels[-1] == -100


def test_char_spans_to_token_labels_multi_word_entity_gets_b_then_i(tiny_tokenizer):
    text = "john smith deposited 100 dollars"
    entities = [
        {"start": 0, "end": 10, "label": "PER", "text": "john smith"},
        {"start": 22, "end": 33, "label": "AMOUNT", "text": "100 dollars"},
    ]
    encoding = char_spans_to_token_labels(text, entities, tiny_tokenizer, label2id=EXTENDED_LABEL2ID)
    word_labels = encoding["word_labels"]
    assert word_labels == ["PER_B", "PER_I", "O", "AMOUNT_B", "AMOUNT_I"]
