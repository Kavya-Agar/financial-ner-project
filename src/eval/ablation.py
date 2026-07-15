"""Before/after ablation: raw model predictions vs. postprocess_entities.

Runs inference once, then computes seqeval metrics twice on the same test
split -- once on the raw predicted BIO labels, once after round-tripping
predictions through `postprocess_entities` -- and prints a per-entity
before/after comparison table.

CLI:

    python -m src.eval.ablation \
        --model-path models/financial-ner-extended \
        --test-file data/synthetic/test.jsonl \
        --output ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.synthetic_data import get_word_spans, word_labels_from_entities
from src.eval.evaluate import (
    bio_labels_to_entities,
    compute_entity_metrics,
    load_jsonl,
    load_model_and_tokenizer,
    predict_word_labels,
)
from src.postprocess.regex_rules import postprocess_entities


def run_ablation(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    id2label: dict[int, str],
    max_length: int | None = None,
) -> dict[str, Any]:
    """Compute raw vs. postprocessed metrics over the same test examples."""
    true_labels_all: list[list[str]] = []
    raw_pred_labels_all: list[list[str]] = []
    post_pred_labels_all: list[list[str]] = []

    for example in examples:
        text = example["text"]
        entities = example.get("entities", [])

        words, word_spans, raw_word_labels = predict_word_labels(
            model, tokenizer, text, id2label, max_length=max_length
        )
        gold_word_labels = word_labels_from_entities(word_spans, entities)

        raw_entities = bio_labels_to_entities(text, word_spans, raw_word_labels)
        cleaned_entities = postprocess_entities(raw_entities)
        post_word_labels = word_labels_from_entities(word_spans, cleaned_entities)

        true_labels_all.append(gold_word_labels)
        raw_pred_labels_all.append(raw_word_labels)
        post_pred_labels_all.append(post_word_labels)

    raw_metrics = compute_entity_metrics(true_labels_all, raw_pred_labels_all)
    post_metrics = compute_entity_metrics(true_labels_all, post_pred_labels_all)

    return {"raw": raw_metrics, "postprocessed": post_metrics}


def ablation_to_markdown(ablation: dict[str, Any]) -> str:
    raw = ablation["raw"]
    post = ablation["postprocessed"]
    labels = sorted(set(raw["per_entity"]) | set(post["per_entity"]))

    lines = [
        "| Entity | Raw P | Raw R | Raw F1 | Post P | Post R | Post F1 | ΔF1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in labels:
        raw_vals = raw["per_entity"].get(label, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        post_vals = post["per_entity"].get(label, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
        delta = post_vals["f1"] - raw_vals["f1"]
        lines.append(
            f"| {label} | {raw_vals['precision']:.3f} | {raw_vals['recall']:.3f} | {raw_vals['f1']:.3f} "
            f"| {post_vals['precision']:.3f} | {post_vals['recall']:.3f} | {post_vals['f1']:.3f} | {delta:+.3f} |"
        )

    raw_micro = raw["micro"]
    post_micro = post["micro"]
    lines.append(
        f"| **micro avg** | {raw_micro['precision']:.3f} | {raw_micro['recall']:.3f} | {raw_micro['f1']:.3f} "
        f"| {post_micro['precision']:.3f} | {post_micro['recall']:.3f} | {post_micro['f1']:.3f} "
        f"| {post_micro['f1'] - raw_micro['f1']:+.3f} |"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw vs. postprocessed model predictions.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--output", type=str, default="ablation.json")
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    examples = load_jsonl(args.test_file)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    ablation = run_ablation(model, tokenizer, examples, id2label, max_length=args.max_length)

    Path(args.output).write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    print(ablation_to_markdown(ablation))
    print(f"\nWrote ablation results to {args.output}")


if __name__ == "__main__":
    main()
