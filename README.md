# Smishing Detection

A four-stage SMS phishing ("smishing") detector: a text model, a URL model and
a sender-reputation lookup, combined by a learned fusion layer that also
explains its verdict.

Reproduction of *Enhancing Mobile Security through Robust Smishing Detection*
(Piyumal & Perera, 2026), with documented deviations — see
[Deviations from the paper](#deviations-from-the-paper).

---

## Quick start

```bash
python -m venv venv
venv/Scripts/activate            # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Score a message:

```bash
python run_pipeline.py "VM-HDFCBK: Your account is suspended. Verify at http://hdfc-secure.tk/login"
python run_pipeline.py --sender VM-HDFCBK "Your account is suspended."
python run_pipeline.py --demo          # built-in sample messages
python run_pipeline.py --json "..."    # raw result dict
```

Or run the demo UI:

```bash
streamlit run app.py
```

> Layer 2 makes a real HTTP request per URL (5s timeout), so a message
> containing links takes a few seconds to analyse.

---

## How a message is scored

```
incoming SMS
   │
   ├─ STAGE 0   src/pipeline/extract.py
   │      Splits the message into sender_id + URLs + body. Catches
   │      schemeless links (bit.ly/xYz, sbi-verify.co.in/login) because
   │      cleaning strips URLs before layer 1 ever sees them — this is the
   │      only place they are captured.
   │
   ├─ LAYER 1   src/pipeline/layer1_sms.py          → smish_probability
   │      clean_pipeline() → TF-IDF → SVM (v3, rbf, C=10).
   │      A message that cleans down to nothing returns "no signal" rather
   │      than a guess; layer 2 still has the link.
   │
   ├─ LAYER 2   src/pipeline/layer2_url.py          → url_phish_score
   │      Wraps src/stage2_url/. Trusted domains short-circuit to 0.0;
   │      otherwise 0.7 × (URL text + lexical model) + 0.3 × live HTML red
   │      flags. Aggregated across links with MAX — a message is only as
   │      safe as its most dangerous link.
   │
   ├─ LAYER 3   src/pipeline/layer3_sender.py       → sender_phish_score
   │      Strict lookup in the custom sender dataset. legit → 0.0,
   │      phishing → 1.0, not found → unknown (None). No heuristics.
   │
   └─ FUSION    src/pipeline/fusion.py              → verdict + reasons
          Logistic regression over the three scores. Absent signals are
          mean-imputed with an explicit presence flag, so "no signal" is
          never read as "safe". Emits a ranked plain-English explanation.
```

Both entry points (`run_pipeline.py`, `app.py`) call `analyze_sms()` in
`src/pipeline/pipeline.py`.

---

## Rebuilding everything from source

The scripts in `scripts/` are numbered in dependency order, with one
exception: **step 12 runs first**. It trains the layer 2 URL model, which step
10 needs in order to score the dataset's URLs. It is numbered 12 only because
01-11 were established before stage 2 was folded in, and renumbering them would
invalidate every existing reference. Read the order as **12 → 01…09 → 10 → 11**.

Run them from the project root:

| # | Script | Produces |
|---|---|---|
| 01 | `01_merge_datasets.py` | `data/processed/combined_raw.csv` |
| 02 | `02_clean_dataset.py` | `data/processed/combined_cleaned.csv` |
| 03 | `03_train_test_split.py` | `train.csv`, `test.csv` (stratified 80/20) |
| 04 | `04_build_tfidf_features.py` | TF-IDF + SVD features, `tfidf_vectorizer.pkl` |
| 05 | `05_build_word2vec_features.py` | Word2Vec model + features |
| 06 | `06_train_ml_models.py` | LR/NB/SVM/RF/XGB on both feature sets → `results/ml_results.json` |
| 07 | `07_augment_data.py` | `train_augmented.csv` (adds transactional SMS) |
| 08 | `08_retrain_svm_v2.py` | `TFIDF_SVM_v2.pkl` |
| 09 | `09_retrain_svm_v3.py` | `TFIDF_SVM_v3.pkl` **(deployed layer 1)** + out-of-fold scores |
| 10 | `10_build_fusion_features.py` | `fusion_features.csv` (+ URL score cache) |
| 11 | `11_train_fusion.py` | `fusion_lr.pkl` **(deployed fusion layer)** |
| 12 | `12_train_stage2_url.py` | `stage2_url_model.pkl` + `stage2_feature_transformer.pkl` **(deployed layer 2)** — runs *before* step 10 |

Steps 04–06 are the model bake-off that selected SVM; the deployed text model
comes from 09. Step 05 is only needed if you want the Word2Vec comparison
numbers.

**Step 12 is provenance, not a step you need to run.** The deployed layer 2
artifacts were fitted under scikit-learn 1.8.0 while this venv runs 1.4.2, so
re-running it will not reproduce them byte for byte — and it overwrites the
deployed model. It is kept so layer 2 is rebuildable and its training data is
documented, the same way layers 1 and 3 are.

`scripts/svd_variance_check.py` is a diagnostic (explained variance vs. number
of SVD components), and `scripts/utils/` holds the one-off dataset download and
verification helpers.

### Layer 2 scores are cached on purpose

Step 10 hits every URL in the dataset over the network. Those scores are cached
per URL in `data/processed/l2_url_scores.csv`, because the fetch is **not**
reproducible — a phishing site that was live during collection is usually dead
months later, and a dead fetch scores differently from a live one. Pass
`--refresh-l2` to re-fetch anyway.

---

## Layout

```
├── app.py                      Streamlit demo (full pipeline)
├── run_pipeline.py             CLI entry point
├── requirements.txt
├── scripts/                    numbered build steps, run in order
│   ├── 01_… 12_…            (12 trains layer 2; see note above)
│   ├── svd_variance_check.py   diagnostic
│   └── utils/                  dataset download + verification helpers,
│                                stage 2 smoke test
├── src/
│   ├── preprocessing/
│   │   └── clean_text.py       clean_pipeline() — used at BOTH train and inference time
│   ├── pipeline/               the four runtime stages + fusion
│   └── stage2_url/             URL model code — features.py + predict.py
│                                (artifacts live in saved_models/)
├── data/
│   ├── raw/                    source datasets, incl. the stage 2 URL
│   │                            training data (stage2_*.csv, 549k rows)
│   ├── processed/              merged, cleaned, split, feature tables
│   ├── features/               .npy feature matrices, Word2Vec model
│   └── custom-dataset_4600.csv.xls   sender dataset (a latin-1 CSV despite the extension)
├── saved_models/               all trained artifacts
├── results/                    ml_results.json
├── notebooks/eda.ipynb
├── figures/
└── archive/                    superseded files — see archive/README.md
```

Only library code lives under `src/`. Anything that runs once to produce an
artifact is a numbered script. The one file that is deliberately both — 
`clean_text.py` — stays in `src/` and is *called* by `scripts/02`, because layer 1
must clean an incoming SMS with exactly the same code the training data went
through.

---

## Deviations from the paper

These are intentional and documented in the source files that implement them:

- **Dataset 1 excluded.** The paper's HuggingFace source turned out to be Enron
  corporate *email* (word cloud dominated by "ect", "hou", "enron", "kaminski"),
  not SMS. See `scripts/01_merge_datasets.py`.
- **Dataset 2 excluded.** The paper cites only "Kaggle" with no name or link;
  it could not be identified with confidence.
- **Dataset 5 added.** A templated persuasion dataset, deduplicated 1000 → 79
  unique messages, to partly offset the class imbalance the two exclusions
  created. Not in the original paper.
- **Per-source deduplication.** Deduplicating the merged set cost Dataset 4 ~82%
  of its rows, because public SMS corpora share verbatim ham messages.
  Per-source dedup matches the paper's reported drop counts far more closely.
- **`class_weight='balanced'`** throughout, for the 83.6/16.4 imbalance.
- **Partial stopword removal.** The paper names words like *verify*, *urgent*,
  *now* as too informative to strip; the full keep-list in
  `src/preprocessing/clean_text.py` extends that with other smishing-signal
  words and is our own addition.
- **Layers 2, 3 and fusion** are not in the paper at all — the paper stops at
  the text model.

## Notes on the fusion layer

Three decisions in `scripts/11_train_fusion.py` deliberately cost headline
accuracy to avoid learning artifacts of how the custom dataset was collected:

- **`l2_present` is excluded as a feature.** In this dataset every phishing row
  carries a URL and no legit-without-URL row is phishing, so "has a link" is a
  perfect one-way rule the model latches onto (coefficient +4.1). Real smishing
  uses callback numbers and reply-bait with no link, and real bank SMS carry
  links constantly. Excluding it costs ~7pp of accuracy and buys a model that
  does not treat "no link" as proof of safety.
- **Layer 3 is trained out-of-fold.** The sender table is rebuilt from the
  training rows of each fold, so held-out rows get an honest "unknown" whenever
  their sender was not seen — which is what happens for the ~55% of senders that
  are not in the table at inference time.
- **Layer 3 is split into two directional indicators** rather than
  (score, present). Out-of-fold there are 194 known-legit matches and *zero*
  known-phishing ones (a phishing sender is single-use, so it is never in the
  training fold when its own row is scored). Sharing one coefficient handed
  known-bad senders an *exonerating* weight; splitting them stops that.
  `l3_known_phish` is all-zero in training and so carries no weight — the data
  honestly cannot say what a known-bad sender is worth.
