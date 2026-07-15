"""Tests for backend/app.py.

Runs fully offline: builds a tiny random-weight DistilBertForTokenClassification
+ BertTokenizerFast (matching the 17-label EXTENDED schema from
src/data/label_schema.py) and points the Flask app's MODEL_DIR at it. No
network access and no real trained model are required.
"""

import json
from pathlib import Path

import pytest
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForTokenClassification

from backend.app import create_app, merge_entities
from src.data.label_schema import EXTENDED_ID2LABEL, EXTENDED_LABEL2ID, NUM_EXTENDED_LABELS

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
VOCAB_PATH = FIXTURES_DIR / "vocab.txt"

ID2LABEL = EXTENDED_ID2LABEL
L2I = EXTENDED_LABEL2ID


# --------------------------------------------------------------------------
# Fixtures: offline tiny model + tokenizer, Flask app/test client
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """Build a tiny random-weight model + tokenizer and save it to disk, so
    the Flask app can load it exactly like it would load a real trained
    model, via AutoTokenizer/AutoModelForTokenClassification.from_pretrained.
    """
    model_dir = tmp_path_factory.mktemp("tiny-ner-model")

    tokenizer = BertTokenizerFast(
        vocab_file=str(VOCAB_PATH),
        do_lower_case=True,
        # Deliberately small so long-text truncation is easy to exercise
        # in tests without needing huge input strings.
        model_max_length=24,
    )

    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=128,
        num_labels=NUM_EXTENDED_LABELS,
        id2label=EXTENDED_ID2LABEL,
        label2id=EXTENDED_LABEL2ID,
    )
    model = DistilBertForTokenClassification(config)
    model.eval()

    tokenizer.save_pretrained(str(model_dir))
    model.save_pretrained(str(model_dir))

    return str(model_dir)


@pytest.fixture()
def app(tiny_model_dir):
    return create_app(model_dir=tiny_model_dir)


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def broken_app(tmp_path):
    """An app pointed at a directory with no model in it, to exercise the
    "model failed to load" paths."""
    missing_dir = tmp_path / "does-not-exist"
    return create_app(model_dir=str(missing_dir))


@pytest.fixture()
def broken_client(broken_app):
    return broken_app.test_client()


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def test_health_ok_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "model_loaded": True}


def test_health_model_not_loaded_still_200(broken_client):
    resp = broken_client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


# --------------------------------------------------------------------------
# / (static frontend build)
# --------------------------------------------------------------------------


def test_index_serves_built_frontend_when_present(tiny_model_dir, tmp_path):
    static_dir = tmp_path / "dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>financial-ner demo</body></html>")

    app = create_app(model_dir=tiny_model_dir, static_dir=str(static_dir))
    resp = app.test_client().get("/")

    assert resp.status_code == 200
    assert b"financial-ner demo" in resp.data


def test_index_404s_when_frontend_not_built(tiny_model_dir, tmp_path):
    static_dir = tmp_path / "dist-not-built"  # deliberately not created

    app = create_app(model_dir=tiny_model_dir, static_dir=str(static_dir))
    resp = app.test_client().get("/")

    assert resp.status_code == 404


# --------------------------------------------------------------------------
# /predict happy path
# --------------------------------------------------------------------------


def test_predict_happy_path_response_shape(client):
    text = "John Smith paid the acme corp bank account."
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200
    data = resp.get_json()

    assert set(data.keys()) == {"text", "entities", "truncated"}
    assert data["text"] == text
    assert isinstance(data["truncated"], bool)
    assert isinstance(data["entities"], list)

    valid_labels = {"PER", "LOC", "ORG", "AMOUNT", "DATE", "ACCOUNT", "SSN", "FORM"}
    starts = []
    for ent in data["entities"]:
        assert set(ent.keys()) == {"text", "label", "start", "end", "score"}
        assert isinstance(ent["start"], int) and isinstance(ent["end"], int)
        assert 0 <= ent["start"] < ent["end"] <= len(text)
        assert ent["text"] == text[ent["start"] : ent["end"]]
        assert ent["label"] in valid_labels
        assert isinstance(ent["score"], float)
        assert 0.0 <= ent["score"] <= 1.0
        starts.append(ent["start"])
    assert starts == sorted(starts)  # sorted by start


# --------------------------------------------------------------------------
# /predict error handling
# --------------------------------------------------------------------------


def test_predict_missing_text_key(client):
    resp = client.post("/predict", json={"foo": "bar"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


@pytest.mark.parametrize(
    "bad_value", [123, 4.5, None, ["a", "b"], {"nested": True}, True]
)
def test_predict_text_wrong_type(client, bad_value):
    resp = client.post("/predict", json={"text": bad_value})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_empty_string_returns_no_entities(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["entities"] == []
    assert data["truncated"] is False


def test_predict_whitespace_only_returns_no_entities(client):
    resp = client.post("/predict", json={"text": "   \n\t  "})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["entities"] == []


def test_predict_very_long_text_is_truncated_not_crashed(client):
    long_text = " ".join(["john smith paid the acme corp bank account"] * 15)
    resp = client.post("/predict", json={"text": long_text})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["truncated"] is True
    assert data["text"] == long_text
    for ent in data["entities"]:
        assert ent["end"] <= len(long_text)


def test_predict_malformed_json_body(client):
    resp = client.post(
        "/predict",
        data="{this is not valid json",
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_wrong_content_type(client):
    resp = client.post(
        "/predict",
        data=json.dumps({"text": "hello world"}),
        content_type="text/plain",
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_model_not_loaded_returns_503(broken_client):
    resp = broken_client.post("/predict", json={"text": "hello world"})
    assert resp.status_code == 503
    assert "error" in resp.get_json()


def test_predict_unicode_offsets_match_original_text(client):
    # Emoji, accented characters, right-to-left Arabic script, and a
    # non-$ currency symbol, all in one string.
    text = "José paid café ₹500 to Müller 😀 مرحبا بالعالم"
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["text"] == text
    assert isinstance(data["entities"], list)
    for ent in data["entities"]:
        # The whole point: char offsets must line up against the *original*
        # text even around multi-byte / astral-plane characters (emoji).
        assert text[ent["start"] : ent["end"]] == ent["text"]


def test_cors_header_present(client):
    # CORS(app) is configured to allow all origins; flask-cors reflects the
    # requesting Origin back rather than emitting a literal "*", which is
    # standard/equivalent behavior for an allow-all policy.
    resp = client.get("/health", headers={"Origin": "http://example.com"})
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://example.com"


# --------------------------------------------------------------------------
# merge_entities: BIO -> character-span merging logic, tested directly with
# synthetic token-level inputs so edge cases are deterministic.
# --------------------------------------------------------------------------


def test_merge_entities_adjacent_b_tags_no_o_between():
    text = "abcdef"
    offset_mapping = [(0, 0), (0, 3), (3, 6), (0, 0)]
    word_ids = [None, 0, 1, None]
    label_ids = [L2I["O"], L2I["PER_B"], L2I["ORG_B"], L2I["O"]]
    scores = [1.0, 0.9, 0.8, 1.0]

    entities = merge_entities(text, offset_mapping, word_ids, label_ids, scores, ID2LABEL)

    assert len(entities) == 2
    assert entities[0] == {"text": "abc", "label": "PER", "start": 0, "end": 3, "score": 0.9}
    assert entities[1] == {"text": "def", "label": "ORG", "start": 3, "end": 6, "score": 0.8}


def test_merge_entities_orphan_i_tag_starts_new_entity():
    text = "12345"
    offset_mapping = [(0, 0), (0, 5), (0, 0)]
    word_ids = [None, 0, None]
    label_ids = [L2I["O"], L2I["AMOUNT_I"], L2I["O"]]
    scores = [1.0, 0.75, 1.0]

    entities = merge_entities(text, offset_mapping, word_ids, label_ids, scores, ID2LABEL)

    assert len(entities) == 1
    assert entities[0]["label"] == "AMOUNT"
    assert entities[0]["start"] == 0
    assert entities[0]["end"] == 5


def test_merge_entities_subword_tokens_not_fragmented():
    text = "Rothschild"
    offset_mapping = [(0, 0), (0, 4), (4, 10), (0, 0)]
    word_ids = [None, 0, 0, None]
    # Second sub-word token deliberately given a *different* label to prove
    # it's ignored: the word must stay a single entity.
    label_ids = [L2I["O"], L2I["ORG_B"], L2I["PER_B"], L2I["O"]]
    scores = [1.0, 0.6, 0.4, 1.0]

    entities = merge_entities(text, offset_mapping, word_ids, label_ids, scores, ID2LABEL)

    assert len(entities) == 1
    assert entities[0] == {"text": "Rothschild", "label": "ORG", "start": 0, "end": 10, "score": 0.6}


def test_merge_entities_special_tokens_excluded_even_if_mislabeled():
    text = "abc"
    offset_mapping = [(0, 0), (0, 3), (0, 0)]
    word_ids = [None, 0, None]
    # CLS/SEP given non-O labels to prove word_ids=None always wins.
    label_ids = [L2I["PER_B"], L2I["O"], L2I["ORG_B"]]
    scores = [0.99, 1.0, 0.99]

    entities = merge_entities(text, offset_mapping, word_ids, label_ids, scores, ID2LABEL)

    assert entities == []


def test_merge_entities_b_i_i_run_merges_into_one_entity():
    text = "New York City"
    offset_mapping = [(0, 0), (0, 3), (4, 8), (9, 13), (0, 0)]
    word_ids = [None, 0, 1, 2, None]
    label_ids = [L2I["O"], L2I["LOC_B"], L2I["LOC_I"], L2I["LOC_I"], L2I["O"]]
    scores = [1.0, 0.9, 0.85, 0.95, 1.0]

    entities = merge_entities(text, offset_mapping, word_ids, label_ids, scores, ID2LABEL)

    assert len(entities) == 1
    ent = entities[0]
    assert ent["label"] == "LOC"
    assert ent["start"] == 0
    assert ent["end"] == 13
    assert ent["text"] == "New York City"
    assert ent["score"] == round((0.9 + 0.85 + 0.95) / 3, 4)


def test_merge_entities_no_entities_when_all_o():
    text = "nothing here"
    offset_mapping = [(0, 0), (0, 7), (8, 12), (0, 0)]
    word_ids = [None, 0, 1, None]
    label_ids = [L2I["O"], L2I["O"], L2I["O"], L2I["O"]]
    scores = [1.0, 1.0, 1.0, 1.0]

    entities = merge_entities(text, offset_mapping, word_ids, label_ids, scores, ID2LABEL)

    assert entities == []
