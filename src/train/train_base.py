"""Fine-tune DistilBERT for token classification on FiNER-ORD (7 base labels).

Reproduces the workflow from `notebooks/legacy/02_main_finetuning.ipynb`
(load FiNER-ORD, group tokens into sentences, tokenize + align labels via
`word_ids()`, train with the HF `Trainer`, evaluate with seqeval) as a
clean, argparse-driven script with no dead code.

Requires network access (to download `gtfintechlab/finer-ord` from the HF
Hub and the `distilbert-base-uncased` weights) -- not runnable in a
network-isolated sandbox, but correct and ready to run in CI or a later
session with network access.

Usage:

    python -m src.train.train_base --epochs 3 --batch-size 32 \
        --output-dir models/financial-ner-v1
"""

from __future__ import annotations

import argparse

import numpy as np
from datasets import Dataset, load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.label_schema import BASE_ID2LABEL, BASE_LABEL2ID, NUM_BASE_LABELS


def group_tokens_by_sentence(dataset_split) -> dict[str, list]:
    """Regroup FiNER-ORD's flat per-token rows into per-sentence token/label lists.

    FiNER-ORD ships one row per token with `doc_idx`/`sent_idx` columns
    identifying which sentence each token belongs to; `gold_token` and
    `gold_label` are the token text and its integer BIO label id.
    """
    df = dataset_split.to_pandas()
    sentences: list[list[str]] = []
    labels: list[list[int]] = []

    for _, group in df.groupby(["doc_idx", "sent_idx"], sort=False):
        group = group.sort_index()
        tokens = []
        token_labels = []
        for token, label in zip(group["gold_token"].tolist(), group["gold_label"].tolist()):
            if token is not None and str(token).strip():
                tokens.append(str(token).strip())
                token_labels.append(label)
        if tokens:
            sentences.append(tokens)
            labels.append(token_labels)

    return {"tokens": sentences, "labels": labels}


def make_tokenize_and_align_labels(tokenizer, max_length: int | None = None):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )
        all_labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id
            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    return tokenize_and_align_labels


def make_compute_metrics(id2label: dict[int, str]):
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

        return {
            "precision": precision_score(true_labels, true_predictions, zero_division=0),
            "recall": recall_score(true_labels, true_predictions, zero_division=0),
            "f1": f1_score(true_labels, true_predictions, zero_division=0),
        }

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on FiNER-ORD (7 base labels).")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--dataset-name", type=str, default="gtfintechlab/finer-ord")
    parser.add_argument("--output-dir", type=str, default="models/financial-ner-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-dir", type=str, default="logs/base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = load_dataset(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_BASE_LABELS,
        id2label=BASE_ID2LABEL,
        label2id=BASE_LABEL2ID,
    )

    train_grouped = group_tokens_by_sentence(dataset["train"])
    val_grouped = group_tokens_by_sentence(dataset["validation"])
    test_grouped = group_tokens_by_sentence(dataset["test"])

    train_dataset = Dataset.from_dict(train_grouped)
    val_dataset = Dataset.from_dict(val_grouped)
    test_dataset = Dataset.from_dict(test_grouped)

    tokenize_and_align_labels = make_tokenize_and_align_labels(tokenizer, max_length=args.max_length)
    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=val_dataset.column_names)
    tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=test_dataset.column_names)

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
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(BASE_ID2LABEL),
    )

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
    print("Test set metrics:", test_metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
