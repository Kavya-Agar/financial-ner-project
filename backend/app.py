"""Flask backend serving the financial NER model.

Loads a token-classification model + tokenizer from ``MODEL_DIR`` (a local
directory produced by ``model.save_pretrained`` / ``tokenizer.save_pretrained``)
and exposes:

    GET  /health   -> {"status": "ok", "model_loaded": bool}
    POST /predict  -> {"text": ..., "entities": [...], "truncated": bool}

The module can be run directly (``python backend/app.py``) or imported by a
WSGI server (``gunicorn backend.app:app``). Tests use ``create_app()``
directly so they can point ``MODEL_DIR`` at an offline fixture without
touching environment variables or process-wide state.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Ensure the repository root (which holds the `src` package) is importable
# regardless of the working directory the app happens to be started from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.label_schema import EXTENDED_ID2LABEL  # noqa: E402

try:
    # The postprocessing/regex-rules module is being built in parallel by
    # another agent. Import and use it rather than reimplementing any of its
    # logic here.
    from src.postprocess.regex_rules import postprocess_entities  # noqa: E402
except ImportError:  # pragma: no cover - exercised only until that module lands
    logging.getLogger(__name__).warning(
        "src.postprocess.regex_rules not found yet; falling back to a "
        "no-op postprocessor. Once that module exists it will be picked up "
        "automatically without any change to this file."
    )

    def postprocess_entities(entities):
        return entities


logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "models/financial-ner-extended"
DEFAULT_MAX_LENGTH = 512


def _resolve_max_length(tokenizer, model) -> int:
    """Pick a sane max sequence length bounding both tokenizer and model."""
    candidates = []
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and tok_max < 1_000_000:
        candidates.append(int(tok_max))
    model_max = getattr(model.config, "max_position_embeddings", None)
    if model_max:
        candidates.append(int(model_max))
    return min(candidates) if candidates else DEFAULT_MAX_LENGTH


def _load_model(model_dir: str):
    """Try to load a tokenizer + model from ``model_dir``.

    Returns ``(tokenizer, model)`` or ``(None, None)`` if loading fails for
    any reason (missing directory, corrupt files, etc). Never raises --
    callers treat a ``(None, None)`` result as "model not loaded" and keep
    serving ``/health`` with ``model_loaded: false``.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        model.eval()
        return tokenizer, model
    except Exception:
        logger.warning("Could not load NER model from %r", model_dir, exc_info=True)
        return None, None


def merge_entities(text, offset_mapping, word_ids, label_ids, scores, id2label):
    """Merge token-level BIO predictions into character-span entities.

    - Special tokens (``[CLS]``/``[SEP]``/``[PAD]``) have a ``word_ids``
      entry of ``None`` and are always skipped, so they can never appear in
      the output.
    - Only the *first* sub-word token of each word carries the label used
      for that word; this guarantees a single word is never fragmented into
      multiple entities purely because of how the tokenizer happened to
      split it into wordpieces.
    - A ``*_B`` tag (or an ``*_I`` tag with no open entity of the same type
      to continue -- e.g. a stray/leading I- tag) always starts a new
      entity rather than crashing or being silently dropped.
    - Two ``*_B`` tags back-to-back (no ``O`` in between) always yield two
      separate entities.
    """
    # First token index + last token end-offset for every word id, in the
    # order words first appear.
    first_token_of_word: dict[int, int] = {}
    last_end_of_word: dict[int, int] = {}
    word_order: list[int] = []
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in first_token_of_word:
            first_token_of_word[wid] = idx
            word_order.append(wid)
        last_end_of_word[wid] = offset_mapping[idx][1]

    entities = []
    current = None
    for wid in word_order:
        idx = first_token_of_word[wid]
        start = offset_mapping[idx][0]
        end = last_end_of_word[wid]
        label = id2label.get(label_ids[idx], "O")
        score = scores[idx]

        if label == "O" or label is None:
            current = None
            continue

        if "_" in label:
            ent_type, bio = label.rsplit("_", 1)
        else:
            # Unexpected label shape; treat as a standalone entity start.
            ent_type, bio = label, "B"

        if bio != "I" or current is None or current["label"] != ent_type:
            current = {"label": ent_type, "start": start, "end": end, "scores": [score]}
            entities.append(current)
        else:
            current["end"] = end
            current["scores"].append(score)

    return [
        {
            "text": text[ent["start"] : ent["end"]],
            "label": ent["label"],
            "start": ent["start"],
            "end": ent["end"],
            "score": round(sum(ent["scores"]) / len(ent["scores"]), 4),
        }
        for ent in entities
    ]


def create_app(model_dir: str | None = None) -> Flask:
    """Application factory.

    ``model_dir`` overrides the ``MODEL_DIR`` env var -- tests use this to
    point at an offline fixture without touching the environment or
    re-importing this module.
    """
    app = Flask(__name__)
    CORS(app)  # portfolio demo: allow all origins

    resolved_model_dir = model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
    tokenizer, model = _load_model(resolved_model_dir)
    model_loaded = tokenizer is not None and model is not None

    if model_loaded:
        raw_id2label = dict(model.config.id2label)
        id2label = {int(k): v for k, v in raw_id2label.items()}
        max_length = _resolve_max_length(tokenizer, model)
    else:
        id2label = dict(EXTENDED_ID2LABEL)
        max_length = DEFAULT_MAX_LENGTH

    app.config["MODEL_DIR"] = resolved_model_dir
    app.config["TOKENIZER"] = tokenizer
    app.config["MODEL"] = model
    app.config["MODEL_LOADED"] = model_loaded
    app.config["ID2LABEL"] = id2label
    app.config["MAX_LENGTH"] = max_length

    @app.get("/health")
    def health():
        try:
            return jsonify({"status": "ok", "model_loaded": bool(app.config.get("MODEL_LOADED"))}), 200
        except Exception:
            # /health must never itself crash the process.
            logger.error("Health check failed unexpectedly", exc_info=True)
            return jsonify({"status": "ok", "model_loaded": False}), 200

    @app.post("/predict")
    def predict():
        if not request.is_json:
            return jsonify({"error": "Request must have Content-Type: application/json."}), 400

        payload = request.get_json(silent=True)
        if payload is None or not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a valid JSON object."}), 400

        if not app.config.get("MODEL_LOADED"):
            return jsonify({"error": "Model is not loaded on the server."}), 503

        if "text" not in payload:
            return jsonify({"error": "Missing required field 'text'."}), 400

        text = payload["text"]
        if not isinstance(text, str):
            return jsonify({"error": "Field 'text' must be a string."}), 400

        if text.strip() == "":
            entities = postprocess_entities([])
            return jsonify({"text": text, "entities": entities, "truncated": False}), 200

        tokenizer = app.config["TOKENIZER"]
        model = app.config["MODEL"]
        id2label = app.config["ID2LABEL"]
        max_length = app.config["MAX_LENGTH"]

        try:
            full_length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            truncated = full_length > max_length

            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            offset_mapping = encoding.pop("offset_mapping")[0].tolist()
            word_ids = encoding.word_ids(0)
            model_inputs = {
                k: v for k, v in encoding.items() if k in ("input_ids", "attention_mask")
            }

            with torch.no_grad():
                logits = model(**model_inputs).logits[0]
            probs = F.softmax(logits, dim=-1)
            token_scores, token_label_ids = probs.max(dim=-1)

            entities = merge_entities(
                text,
                offset_mapping,
                word_ids,
                token_label_ids.tolist(),
                token_scores.tolist(),
                id2label,
            )
            entities = postprocess_entities(entities)
            entities.sort(key=lambda e: e["start"])

            return jsonify({"text": text, "entities": entities, "truncated": truncated}), 200
        except Exception:
            logger.error("Prediction failed", exc_info=True)
            return jsonify({"error": "Internal error while running inference."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
