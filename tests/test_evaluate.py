"""Tests for src/eval/evaluate.py.

`compute_entity_metrics` is pure (hand-constructed label sequences, no
model/network). `bio_labels_to_entities` and `predict_word_labels` are
exercised with the tiny offline tokenizer/model fixtures.
"""

import pytest

from src.eval.evaluate import bio_labels_to_entities, compute_entity_metrics, predict_word_labels

# ---------------------------------------------------------------------------
# compute_entity_metrics: pure seqeval wrapper
# ---------------------------------------------------------------------------


def test_compute_entity_metrics_perfect_match():
    true = [["O", "PER_B", "PER_I", "O"]]
    pred = [["O", "PER_B", "PER_I", "O"]]
    metrics = compute_entity_metrics(true, pred)
    assert metrics["micro"]["precision"] == 1.0
    assert metrics["micro"]["recall"] == 1.0
    assert metrics["micro"]["f1"] == 1.0
    assert metrics["per_entity"]["PER"]["f1"] == 1.0


def test_compute_entity_metrics_all_wrong():
    true = [["O", "PER_B", "PER_I", "O"]]
    pred = [["O", "O", "O", "O"]]
    metrics = compute_entity_metrics(true, pred)
    assert metrics["micro"]["recall"] == 0.0
    assert metrics["micro"]["f1"] == 0.0


def test_compute_entity_metrics_partial_boundary_mismatch():
    # Gold entity spans two words; prediction only catches the first word.
    true = [["PER_B", "PER_I", "O"]]
    pred = [["PER_B", "O", "O"]]
    metrics = compute_entity_metrics(true, pred)
    # seqeval is entity-level: a boundary mismatch counts as both a false
    # positive and a false negative for that entity type, not a partial credit.
    assert metrics["micro"]["f1"] == 0.0
    assert metrics["per_entity"]["PER"]["support"] == 1


def test_compute_entity_metrics_empty_prediction_list():
    true = [[]]
    pred = [[]]
    metrics = compute_entity_metrics(true, pred)
    assert metrics["per_entity"] == {}
    assert metrics["micro"]["f1"] == 0.0


def test_compute_entity_metrics_no_sequences_at_all():
    metrics = compute_entity_metrics([], [])
    assert metrics["per_entity"] == {}
    assert metrics["micro"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_compute_entity_metrics_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_entity_metrics([["O"]], [["O"], ["O"]])


def test_compute_entity_metrics_multiple_entity_types():
    true = [["PER_B", "O", "AMOUNT_B", "AMOUNT_I"]]
    pred = [["PER_B", "O", "AMOUNT_B", "AMOUNT_I"]]
    metrics = compute_entity_metrics(true, pred)
    assert set(metrics["per_entity"].keys()) == {"PER", "AMOUNT"}
    assert metrics["macro_f1"] == 1.0


# ---------------------------------------------------------------------------
# bio_labels_to_entities: reconstruct spans from word-level BIO labels
# ---------------------------------------------------------------------------


def test_bio_labels_to_entities_basic():
    text = "John Smith deposited 100 dollars"
    word_spans = [(0, 4), (5, 10), (11, 20), (21, 24), (25, 33)]
    word_labels = ["PER_B", "PER_I", "O", "AMOUNT_B", "AMOUNT_I"]
    entities = bio_labels_to_entities(text, word_spans, word_labels)
    assert entities == [
        {"start": 0, "end": 10, "label": "PER", "text": "John Smith"},
        {"start": 21, "end": 33, "label": "AMOUNT", "text": "100 dollars"},
    ]


def test_bio_labels_to_entities_handles_i_without_preceding_b():
    # Model mis-predicted an I tag with no B before it; should still start a
    # new entity rather than crashing or silently dropping it.
    text = "smith deposited"
    word_spans = [(0, 5), (6, 15)]
    word_labels = ["PER_I", "O"]
    entities = bio_labels_to_entities(text, word_spans, word_labels)
    assert entities == [{"start": 0, "end": 5, "label": "PER", "text": "smith"}]


def test_bio_labels_to_entities_empty_labels():
    assert bio_labels_to_entities("", [], []) == []


def test_bio_labels_to_entities_entity_runs_to_end_of_sequence():
    text = "paid john smith"
    word_spans = [(0, 4), (5, 9), (10, 15)]
    word_labels = ["O", "PER_B", "PER_I"]
    entities = bio_labels_to_entities(text, word_spans, word_labels)
    assert entities == [{"start": 5, "end": 15, "label": "PER", "text": "john smith"}]


# ---------------------------------------------------------------------------
# predict_word_labels: exercised end-to-end with tiny offline fixtures
# ---------------------------------------------------------------------------


def test_predict_word_labels_runs_and_returns_one_label_per_word(tiny_tokenizer, tiny_model_factory):
    id2label = {i: label for i, label in enumerate(
        ["O", "PER_B", "PER_I", "LOC_B", "LOC_I", "ORG_B", "ORG_I"]
    )}
    model = tiny_model_factory(num_labels=7, id2label=id2label, label2id={v: k for k, v in id2label.items()}, seed=0)

    text = "john smith deposited 100 dollars"
    words, word_spans, pred_labels = predict_word_labels(model, tiny_tokenizer, text, id2label)

    assert len(words) == len(word_spans) == len(pred_labels)
    assert all(label in id2label.values() for label in pred_labels)


def test_predict_word_labels_empty_text(tiny_tokenizer, tiny_model_factory):
    id2label = {0: "O", 1: "PER_B"}
    model = tiny_model_factory(num_labels=2, id2label=id2label, label2id={v: k for k, v in id2label.items()}, seed=0)
    words, word_spans, pred_labels = predict_word_labels(model, tiny_tokenizer, "", id2label)
    assert words == []
    assert pred_labels == []
