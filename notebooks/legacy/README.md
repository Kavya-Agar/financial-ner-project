## Legacy notebooks

These four notebooks are the original exploratory work and are kept for
history only — they are **not** the current training/eval pipeline.

- `01_quickstart_test.ipynb` — environment setup / sanity checks
- `02_main_finetuning.ipynb` — first fine-tune of DistilBERT on FiNER-ORD (7 labels)
- `03_entity_extension.ipynb` — first attempt at extending to 17 labels via synthetic data
- `04_testing_model.ipynb` — manual spot-checking of inference + regex post-processing

The reproducible, tested version of this pipeline lives in `src/`. In
particular, `03_entity_extension.ipynb` did **not** actually transfer the
trained 7-label weights into the 17-label model (it fell back to a fresh
`distilbert-base-uncased` init) — `src/train/extend_model.py` fixes that.
