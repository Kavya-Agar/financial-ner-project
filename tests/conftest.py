"""Shared pytest fixtures: tiny offline tokenizer + model, no network needed.

The tokenizer is built from a local vocab file
(tests/fixtures/ner_vocab.txt, distinct from tests/fixtures/vocab.txt
which belongs to the pre-existing backend test suite) via
`BertTokenizerFast(vocab=...)` -- a local path, so this never touches the
network. The model is a small `DistilBertForTokenClassification`
constructed directly from a `DistilBertConfig` (never `from_pretrained`
with a model name), so it also never touches the network.
"""

from pathlib import Path

import pytest
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForTokenClassification

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_VOCAB_PATH = _FIXTURES_DIR / "ner_vocab.txt"


def _vocab_size() -> int:
    with _VOCAB_PATH.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return BertTokenizerFast(vocab=str(_VOCAB_PATH), do_lower_case=True)


@pytest.fixture()
def tiny_model_factory():
    """Returns a factory `make_model(num_labels, id2label=None, label2id=None, seed=0)`."""

    def make_model(num_labels: int, id2label=None, label2id=None, seed: int = 0):
        import torch

        torch.manual_seed(seed)
        config = DistilBertConfig(
            vocab_size=_vocab_size(),
            dim=32,
            n_layers=2,
            n_heads=2,
            hidden_dim=64,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        return DistilBertForTokenClassification(config)

    return make_model
