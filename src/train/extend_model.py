"""Extend the 7-label FiNER-ORD model to the 17-label financial entity set.

This is the fix for `notebooks/legacy/03_entity_extension.ipynb`, which
never actually transferred the trained 7-label weights into the 17-label
model (its code comment admits: "For now, we'll start fresh"). Here, the
transfer is real:

  * The entire DistilBERT encoder body is copied from the trained base
    model into the extended model (not re-downloaded/re-initialized).
  * The classification head's first `NUM_BASE_LABELS` rows (PER/LOC/ORG,
    in the same row order as the base model -- see
    `src/data/label_schema.py`) are copied from the base model's head.
  * Rows `NUM_BASE_LABELS:NUM_EXTENDED_LABELS` (AMOUNT/DATE/ACCOUNT/SSN/
    FORM) are left at their random initialization, to be learned from the
    synthetic data.

Requires network access (to build a fresh randomly-initialized
DistilBERT-shaped model config/architecture reference) -- specifically,
only the *first* run needs `distilbert-base-uncased`'s config; if the base
model directory already has a full local config it needs none. Continued
training itself needs no network. Not runnable in a network-isolated
sandbox if `models/financial-ner-v1` doesn't exist yet, but correct and
ready to run in CI or a later session.

Usage:

    python -m src.train.extend_model \
        --base-model-path models/financial-ner-v1 \
        --synthetic-data-dir data/synthetic \
        --output-dir models/financial-ner-extended
"""

from __future__ import annotations

import argparse

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.label_schema import (
    EXTENDED_ID2LABEL,
    EXTENDED_LABEL2ID,
    NUM_BASE_LABELS,
    NUM_EXTENDED_LABELS,
)
from src.data.synthetic_data import char_spans_to_token_labels
from src.eval.evaluate import compute_entity_metrics, load_jsonl


def transfer_classifier_weights(base_model, extended_model, num_base_labels: int):
    """Copy backbone + head weights from `base_model` into `extended_model`, in place.

    * The full encoder body (`.distilbert`) is copied wholesale.
    * Classifier rows `[0:num_base_labels]` are copied from the base
      model's classifier weight/bias.
    * Classifier rows `[num_base_labels:]` are left untouched (the
      extended model's own random initialization).

    Raises `ValueError` with a clear message if the two models' hidden
    sizes don't match (which would otherwise surface as a confusing
    shape-mismatch error deep inside `load_state_dict`).

    Returns `extended_model` (mutated in place) for convenience.
    """
    base_hidden_size = base_model.classifier.weight.shape[1]
    extended_hidden_size = extended_model.classifier.weight.shape[1]
    if base_hidden_size != extended_hidden_size:
        raise ValueError(
            "Cannot transfer classifier weights: base model hidden size "
            f"({base_hidden_size}) does not match extended model hidden size "
            f"({extended_hidden_size}). The two models must share the same "
            "DistilBERT architecture (dim/n_heads/hidden_dim) for a row-wise "
            "weight copy to be valid."
        )

    extended_num_labels = extended_model.classifier.weight.shape[0]
    if num_base_labels > extended_num_labels:
        raise ValueError(
            f"num_base_labels ({num_base_labels}) exceeds the extended model's "
            f"number of output labels ({extended_num_labels})."
        )

    extended_model.distilbert.load_state_dict(base_model.distilbert.state_dict())

    with torch.no_grad():
        extended_model.classifier.weight[:num_base_labels] = base_model.classifier.weight[:num_base_labels]
        extended_model.classifier.bias[:num_base_labels] = base_model.classifier.bias[:num_base_labels]

    return extended_model


def build_extended_model(base_model_path: str):
    """Load the trained base model and build a fresh extended-label model
    with the same encoder architecture, ready for weight transfer."""
    base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    extended_config = AutoConfig.from_pretrained(
        base_model_path,
        num_labels=NUM_EXTENDED_LABELS,
        id2label=EXTENDED_ID2LABEL,
        label2id=EXTENDED_LABEL2ID,
    )
    extended_model = AutoModelForTokenClassification.from_config(extended_config)

    transfer_classifier_weights(base_model, extended_model, NUM_BASE_LABELS)
    return extended_model, tokenizer


def build_synthetic_dataset(jsonl_path: str, tokenizer, max_length: int | None = None) -> Dataset:
    examples = load_jsonl(jsonl_path)
    all_input_ids, all_attention_mask, all_labels = [], [], []
    for example in examples:
        encoding = char_spans_to_token_labels(
            example["text"], example["entities"], tokenizer, label2id=EXTENDED_LABEL2ID, max_length=max_length
        )
        all_input_ids.append(encoding["input_ids"])
        all_attention_mask.append(encoding["attention_mask"])
        all_labels.append(encoding["labels"])
    return Dataset.from_dict({"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels})


def make_compute_metrics(id2label: dict[int, str]):
    import numpy as np

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for p, l in zip(pred_seq, label_seq) if l != -100]
            for pred_seq, label_seq in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for p, l in zip(pred_seq, label_seq) if l != -100]
            for pred_seq, label_seq in zip(predictions, labels)
        ]
        metrics = compute_entity_metrics(true_labels, true_predictions)
        return {
            "precision": metrics["micro"]["precision"],
            "recall": metrics["micro"]["recall"],
            "f1": metrics["micro"]["f1"],
            "macro_f1": metrics["macro_f1"],
        }

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend a 7-label financial NER model to 17 labels via synthetic data.")
    parser.add_argument("--base-model-path", type=str, default="models/financial-ner-v1")
    parser.add_argument("--synthetic-data-dir", type=str, default="data/synthetic")
    parser.add_argument("--output-dir", type=str, default="models/financial-ner-extended")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-dir", type=str, default="logs/extended")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    extended_model, tokenizer = build_extended_model(args.base_model_path)

    train_dataset = build_synthetic_dataset(f"{args.synthetic_data_dir}/train.jsonl", tokenizer, max_length=args.max_length)
    val_dataset = build_synthetic_dataset(f"{args.synthetic_data_dir}/val.jsonl", tokenizer, max_length=args.max_length)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_dir=args.logging_dir,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=extended_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(EXTENDED_ID2LABEL),
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved extended model + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
