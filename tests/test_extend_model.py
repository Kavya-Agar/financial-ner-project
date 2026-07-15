"""Tests for src/train/extend_model.py's transfer_classifier_weights, the
actual fix for the legacy notebook's "start fresh" bug. Uses tiny offline
DistilBertForTokenClassification fixtures (no network)."""

import pytest
import torch

from src.train.extend_model import transfer_classifier_weights


def test_transfer_copies_base_rows_exactly(tiny_model_factory):
    base_model = tiny_model_factory(num_labels=7, seed=1)
    extended_model = tiny_model_factory(num_labels=17, seed=2)

    transfer_classifier_weights(base_model, extended_model, num_base_labels=7)

    assert torch.equal(extended_model.classifier.weight[:7], base_model.classifier.weight[:7])
    assert torch.equal(extended_model.classifier.bias[:7], base_model.classifier.bias[:7])


def test_transfer_leaves_new_rows_at_extended_models_own_init(tiny_model_factory):
    base_model = tiny_model_factory(num_labels=7, seed=1)
    extended_model = tiny_model_factory(num_labels=17, seed=2)

    # Snapshot the extended model's own random init for rows [7:17] before transfer.
    original_new_weight = extended_model.classifier.weight[7:].clone()
    original_new_bias = extended_model.classifier.bias[7:].clone()

    transfer_classifier_weights(base_model, extended_model, num_base_labels=7)

    assert torch.equal(extended_model.classifier.weight[7:], original_new_weight)
    assert torch.equal(extended_model.classifier.bias[7:], original_new_bias)


def test_transfer_copies_encoder_body(tiny_model_factory):
    base_model = tiny_model_factory(num_labels=7, seed=1)
    extended_model = tiny_model_factory(num_labels=17, seed=2)

    # Before transfer, encoder weights differ (different seeds).
    base_layer0_weight = base_model.distilbert.transformer.layer[0].attention.q_lin.weight
    extended_layer0_weight_before = extended_model.distilbert.transformer.layer[0].attention.q_lin.weight.clone()
    assert not torch.equal(base_layer0_weight, extended_layer0_weight_before)

    transfer_classifier_weights(base_model, extended_model, num_base_labels=7)

    extended_layer0_weight_after = extended_model.distilbert.transformer.layer[0].attention.q_lin.weight
    assert torch.equal(base_layer0_weight, extended_layer0_weight_after)


def test_transfer_returns_the_extended_model(tiny_model_factory):
    base_model = tiny_model_factory(num_labels=7, seed=1)
    extended_model = tiny_model_factory(num_labels=17, seed=2)
    result = transfer_classifier_weights(base_model, extended_model, num_base_labels=7)
    assert result is extended_model


def test_transfer_raises_clear_error_on_hidden_size_mismatch(tiny_model_factory):
    import torch as _torch
    from transformers import DistilBertConfig, DistilBertForTokenClassification

    base_model = tiny_model_factory(num_labels=7, seed=1)

    _torch.manual_seed(3)
    mismatched_config = DistilBertConfig(
        vocab_size=67,
        dim=48,  # different hidden size than the tiny_model_factory's dim=32
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        num_labels=17,
    )
    mismatched_extended_model = DistilBertForTokenClassification(mismatched_config)

    with pytest.raises(ValueError, match="hidden size"):
        transfer_classifier_weights(base_model, mismatched_extended_model, num_base_labels=7)


def test_transfer_raises_when_num_base_labels_exceeds_extended_labels(tiny_model_factory):
    base_model = tiny_model_factory(num_labels=7, seed=1)
    extended_model = tiny_model_factory(num_labels=5, seed=2)

    with pytest.raises(ValueError):
        transfer_classifier_weights(base_model, extended_model, num_base_labels=7)
