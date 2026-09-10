# archive/

Superseded files, kept only so nothing from the build history is lost. Nothing
here is imported, trained on, or run by the current pipeline — deleting this
whole folder would not affect the project.

| File | What it was | Why it was superseded |
|---|---|---|
| `app_w2v_single_model.py` | The first Streamlit demo (was `src/preprocessing/app.py`). Loaded `W2V_SVM.pkl` + `w2v_scaler.pkl` and classified a message with that one model. | Replaced by the root `app.py`, which runs the full four-stage pipeline (extract → text → URL → sender → fusion) instead of a single text model. Also had a broken `sys.path` line and CWD-relative model paths, so it only ran from the project root. |
| `LR_TFIDF.pkl`, `NB_TFIDF.pkl` | Two models from an early run, written to a `models/` directory. | `scripts/06_train_ml_models.py` writes to `saved_models/`, where the same two models exist as `TFIDF_LR.pkl` and `TFIDF_NB.pkl`. The `models/` directory was a leftover from before the output path settled. |
| `stage2_model_5feature.pkl`, `stage2_vectorizer_5feature.pkl` | An earlier stage 2 URL model (`model.pkl` / `vectorizer.pkl`), loaded only by `app_stage2_url_only.py`. Built on **5** lexical features. | The deployed model uses **7** (adds `url_length`, `domain_length`) and is a different fit entirely — the md5s match nothing in `saved_models/`. Nothing regenerates these; `scripts/12_train_stage2_url.py` produces the 7-feature pair. |
| `app_stage2_url_only.py` | The standalone Streamlit demo shipped with the stage 2 URL model (was `Website Phishing/app.py`). Scored a URL with the 5-feature model and a hand-rolled red-flag count. | Superseded by `src/stage2_url/predict.py`, which the pipeline calls: 7 lexical features and a documented `0.7 × text/lexical + 0.3 × HTML red flags` blend instead of ad-hoc `if red_flags >= 2` thresholds. Also had CWD-relative model paths. |
| `requirements_stage2_freeze.txt` | Shipped alongside the stage 2 URL model as its requirements file. | Not a requirements file for stage 2 — it is a UTF-16 `pip freeze` of an unrelated environment (torch, transformers, jupyter, flask). Its one useful fact, that the stage 2 pickles were produced under **scikit-learn 1.8.0**, is now recorded in `requirements.txt` and in the shim in `src/pipeline/layer2_url.py`. |

Also removed during the same cleanup, with nothing to preserve:

- `main.py` — 0 bytes, never written.
- `eda.ipynb` (project root) — 0 bytes; the real notebook is `notebooks/eda.ipynb`.
- `logs/` — empty directory, nothing ever logged to it.
- `Website Phishing/requirements_stage2.txt` — byte-identical duplicate of
  `requirements_stage2_freeze.txt` above (same md5), so only the one copy is kept.

The `Website Phishing/` folder that these came from was the stage 2 URL model's
original standalone project. Its still-current parts were distributed into the
tree — datasets to `data/raw/stage2_*.csv`, `train_model.py` to
`scripts/12_train_stage2_url.py`, `test_predict.py` to
`scripts/utils/test_stage2_predict.py`, and the two deployed pickles to
`saved_models/` — and the folder itself was removed. Its `features.py` was
byte-identical to `src/stage2_url/features.py`, and its `predict.py` differed
only in resolving model paths relative to the CWD.
