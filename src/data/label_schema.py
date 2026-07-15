"""Shared BIO label schema for the financial NER model.

The first 7 labels (indices 0-6) match the base model trained on FiNER-ORD.
Indices 7-16 are the financial-document entities added on top. This ordering
is load-bearing: src/train/extend_model.py copies classification-head weights
by row index from the 7-label model into the first 7 rows of the 17-label
model, so PER/LOC/ORG must keep the same indices in both label sets.
"""

BASE_ENTITY_TYPES = ["PER", "LOC", "ORG"]
EXTENDED_ENTITY_TYPES = ["AMOUNT", "DATE", "ACCOUNT", "SSN", "FORM"]
ALL_ENTITY_TYPES = BASE_ENTITY_TYPES + EXTENDED_ENTITY_TYPES


def _bio_labels(entity_types):
    labels = ["O"]
    for ent in entity_types:
        labels.append(f"{ent}_B")
        labels.append(f"{ent}_I")
    return labels


BASE_LABELS = _bio_labels(BASE_ENTITY_TYPES)
EXTENDED_LABELS = _bio_labels(ALL_ENTITY_TYPES)

BASE_LABEL2ID = {label: i for i, label in enumerate(BASE_LABELS)}
BASE_ID2LABEL = {i: label for label, i in BASE_LABEL2ID.items()}

EXTENDED_LABEL2ID = {label: i for i, label in enumerate(EXTENDED_LABELS)}
EXTENDED_ID2LABEL = {i: label for label, i in EXTENDED_LABEL2ID.items()}

NUM_BASE_LABELS = len(BASE_LABELS)
NUM_EXTENDED_LABELS = len(EXTENDED_LABELS)

assert BASE_LABELS == EXTENDED_LABELS[:NUM_BASE_LABELS], (
    "Base labels must be a prefix of extended labels for weight-copy transfer "
    "learning to line up by row index."
)
