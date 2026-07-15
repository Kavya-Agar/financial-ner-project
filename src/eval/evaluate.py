"""Evaluate a trained token-classification model on a synthetic test split.

Deliberately split into two layers:

  1. Inference (`predict_examples`) -- needs a loaded model + tokenizer.
  2. Metrics (`compute_entity_metrics`) -- pure function over gold/predicted
     BIO label sequences, no model or network required. This is the part
     that's unit tested with hand-constructed label sequences.

CLI:

    python -m src.eval.evaluate \
        --model-path models/financial-ner-extended \
        --test-file data/synthetic/test.jsonl \
        --output metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from seqeval.metrics import (
    classification_report as seqeval_classification_report,
)
from seqeval.metrics import f1_score, precision_score, recall_score

from src.data.synthetic_data import get_word_spans, word_labels_from_entities

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_model_and_tokenizer(model_path: str):
    """Load a token-classification model + its tokenizer from a local path.

    Requires the model to already exist on disk (produced by
    `src/train/train_base.py` or `src/train/extend_model.py`); does not
    need network access for a local path.
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_word_labels(
    model,
    tokenizer,
    text: str,
    id2label: dict[int, str],
    max_length: int | None = None,
) -> tuple[list[str], list[tuple[int, int]], list[str]]:
    """Run the model on `text` and return one predicted BIO label per word.

    Returns (words, word_spans, predicted_word_labels). Uses the tokenizer's
    own pre-tokenizer (via `get_word_spans`) so word boundaries exactly
    match how the model was trained, and `word_ids()` to pick the first
    subword's prediction as the word-level label (the same convention used
    to build the training labels).
    """
    import torch

    words, word_spans = get_word_spans(text, tokenizer)
    if not words:
        return words, word_spans, []

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)

    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        ).logits
    predicted_ids = logits.argmax(dim=-1)[0].tolist()

    pred_word_labels = ["O"] * len(words)
    seen_words: set[int] = set()
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_words:
            continue
        seen_words.add(word_id)
        pred_word_labels[word_id] = id2label[predicted_ids[token_idx]]

    return words, word_spans, pred_word_labels


def predict_examples(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    id2label: dict[int, str],
    max_length: int | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Run inference over a list of {"text", "entities"} examples.

    Returns (true_label_sequences, pred_label_sequences), ready to hand to
    `compute_entity_metrics`.
    """
    true_labels_all: list[list[str]] = []
    pred_labels_all: list[list[str]] = []

    for example in examples:
        text = example["text"]
        entities = example.get("entities", [])
        words, word_spans, pred_word_labels = predict_word_labels(
            model, tokenizer, text, id2label, max_length=max_length
        )
        gold_word_labels = word_labels_from_entities(word_spans, entities)
        true_labels_all.append(gold_word_labels)
        pred_labels_all.append(pred_word_labels)

    return true_labels_all, pred_labels_all


def bio_labels_to_entities(
    text: str, word_spans: list[tuple[int, int]], word_labels: list[str]
) -> list[dict[str, Any]]:
    """Reconstruct entity spans from word-level BIO labels + char offsets.

    Used by `src/eval/ablation.py` to turn raw predictions into entities
    that `postprocess_entities` can operate on, then converted back to BIO
    labels for a like-for-like seqeval comparison.
    """
    entities: list[dict[str, Any]] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    for (w_start, w_end), label in zip(word_spans, word_labels):
        if not label or label == "O":
            if current_label is not None:
                entities.append(
                    {"start": current_start, "end": current_end, "label": current_label, "text": text[current_start:current_end]}
                )
                current_label = None
            continue

        ent_type, _, bio = label.rpartition("_")
        if bio == "B" or ent_type != current_label:
            if current_label is not None:
                entities.append(
                    {"start": current_start, "end": current_end, "label": current_label, "text": text[current_start:current_end]}
                )
            current_label = ent_type
            current_start = w_start
            current_end = w_end
        else:
            current_end = w_end

    if current_label is not None:
        entities.append(
            {"start": current_start, "end": current_end, "label": current_label, "text": text[current_start:current_end]}
        )

    return entities


# ---------------------------------------------------------------------------
# Metrics (pure, no model/network needed)
# ---------------------------------------------------------------------------


def _to_seqeval_scheme(label_sequences: list[list[str]]) -> list[list[str]]:
    """Convert our "TYPE_B"/"TYPE_I" suffix labels to seqeval's expected
    "B-TYPE"/"I-TYPE" prefix (IOB2) notation.

    `src/data/label_schema.py` uses the suffix convention throughout (e.g.
    "PER_B"), but seqeval's chunk extraction only recognizes the standard
    "B-PER"/"I-PER" prefix convention -- passing suffix-style labels
    straight through silently produces garbage entity types (it strips
    what it assumes is a "B-" prefix, mangling "PER_B" into "ER_B").
    """
    converted = []
    for sequence in label_sequences:
        new_sequence = []
        for label in sequence:
            if label == "O" or not label:
                new_sequence.append("O")
                continue
            entity_type, _, bio = label.rpartition("_")
            new_sequence.append(f"{bio}-{entity_type}" if entity_type else label)
        converted.append(new_sequence)
    return converted


def compute_entity_metrics(
    true_labels: list[list[str]], pred_labels: list[list[str]]
) -> dict[str, Any]:
    """Compute per-entity and overall precision/recall/F1 via seqeval.

    `true_labels`/`pred_labels` are parallel lists of BIO label sequences
    in our "TYPE_B"/"TYPE_I" suffix notation (one sequence per example).
    Raises `ValueError` on malformed input (mismatched lengths) but
    otherwise handles empty sequences gracefully.
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError(
            f"true_labels and pred_labels must have the same number of sequences "
            f"(got {len(true_labels)} and {len(pred_labels)})"
        )
    if not true_labels:
        return {"per_entity": {}, "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "macro_f1": 0.0}

    true_labels = _to_seqeval_scheme(true_labels)
    pred_labels = _to_seqeval_scheme(pred_labels)

    report = seqeval_classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

    per_entity = {
        label: {
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1": vals["f1-score"],
            "support": vals["support"],
        }
        for label, vals in report.items()
        if label not in ("micro avg", "macro avg", "weighted avg")
    }

    micro = {
        "precision": precision_score(true_labels, pred_labels, average="micro", zero_division=0),
        "recall": recall_score(true_labels, pred_labels, average="micro", zero_division=0),
        "f1": f1_score(true_labels, pred_labels, average="micro", zero_division=0),
    }
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    return {"per_entity": per_entity, "micro": micro, "macro_f1": macro_f1}


def metrics_to_markdown(metrics: dict[str, Any]) -> str:
    lines = ["| Entity | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for label, vals in sorted(metrics["per_entity"].items()):
        lines.append(
            f"| {label} | {vals['precision']:.3f} | {vals['recall']:.3f} | {vals['f1']:.3f} | {vals['support']} |"
        )
    micro = metrics["micro"]
    lines.append(f"| **micro avg** | {micro['precision']:.3f} | {micro['recall']:.3f} | {micro['f1']:.3f} | - |")
    lines.append(f"| **macro F1** | - | - | {metrics['macro_f1']:.3f} | - |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a token-classification model on a test split.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--output", type=str, default="metrics.json")
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    examples = load_jsonl(args.test_file)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    true_labels, pred_labels = predict_examples(model, tokenizer, examples, id2label, max_length=args.max_length)
    metrics = compute_entity_metrics(true_labels, pred_labels)

    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics_to_markdown(metrics))
    print(f"\nWrote metrics to {args.output}")


if __name__ == "__main__":
    main()
