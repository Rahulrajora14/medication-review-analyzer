# 📘 How this project works

A complete walkthrough of the codebase: what each file does, why each decision was made, and how to explain it in an interview. Read it once top to bottom, then use it as a reference.

---

## 1. The big picture in one minute

**Question the project answers:** *"What are patients saying about their medicines: are they happy, and if not, what side effects are they complaining about?"*

**The flow:**

```
2 TSV files (215k reviews)
      │
      ▼
prepare.py ── clean ── label ── dedupe ── remove leaked test rows ── split
      │
      ├──► eda.py ──► charts + per-drug side-effect tables ──┐
      │                                                        │
      ├──► baseline.py (TF-IDF + LogReg, CPU, 3 min) ──┐       │
      │                                                │       ├──► Streamlit app
      └──► transformer.py (DistilBERT, GPU, ~40 min) ──┤       │
                                                       ▼       │
                                 report.py ──► error_analysis.py
```

**One command runs it:** `python main.py`. DistilBERT runs only when a GPU is present; otherwise you use the Kaggle notebook.

**How to describe it in 30 seconds (memorise this):**

> "I built an NLP pipeline on 215k real patient drug reviews. While preparing the data I found that over 32,000 of the official test reviews were exact copies of training reviews, because the same review is listed under brand and generic drug names, so I removed them to get an honest test set. I trained a TF-IDF and Logistic Regression baseline, then fine-tuned DistilBERT, a 66-million-parameter small language model, with my own PyTorch training loop using AdamW, warmup scheduling, mixed precision and class-weighted loss. Error analysis showed the hard cases are mixed reviews like 'it works, but…' and borderline ratings. I also built a condition-aware side-effect extractor and a Streamlit dashboard that shows each drug's side-effect profile."

---

## 2. File-by-file walkthrough

Read in this order; it follows the pipeline.

### `src/config.py`: the control panel

Every path and setting lives here: label boundaries, TF-IDF size, DistilBERT learning rate, batch size, epochs. Every other file does `from src import config`.

The settings worth knowing by heart: `MAX_LENGTH = 256` tokens, `BATCH_SIZE = 32`, `EPOCHS = 2`, `LEARNING_RATE = 3e-5`, `WARMUP_RATIO = 0.06`. `TRAIN_SUBSET` lets you train on fewer reviews for a quicker run.

---

### `src/data/download_data.py`: step 1

Downloads the UCI zip and extracts the two `.tsv` files (tab-separated, because reviews contain commas). If UCI fails, it uses a GitHub mirror. If the files already exist, it skips.

---

### `src/data/prepare.py`: step 2 (read this file carefully)

| Function | What it does |
|---|---|
| `clean_text()` | `html.unescape` turns `&#039;` into `'` and `&amp;` into `&`; strips the wrapping quotes; collapses repeated spaces |
| `rating_to_label()` | 1–4 → 0 (negative), 5–6 → 1 (neutral), 7–10 → 2 (positive) |
| `clean_frame()` | applies cleaning, fixes broken `condition` values (scraped HTML like `3</span> users found…`), parses dates, adds `label`, `sentiment`, `n_words` |
| `prepare_data()` | de-duplication, leakage fix, validation split, saves `.parquet` files |
| `load_split()` | used by every other step to read `train` / `val` / `test` |

**The two de-duplication rules (your best interview story):**

1. **Within each file, keep one copy of each review text.** Drugs.com lists the same review under the brand name and the generic name (*Nexplanon* and *Etonogestrel* are the same implant). Train drops from 161,297 to 112,312.
2. **Remove every test review whose text also appears in train.** 32,131 removed. Without this, the model is tested on reviews it memorised during training, so the test score measures memory, not understanding.

**Why parquet instead of CSV?** It's smaller, much faster to load, and keeps data types (dates stay dates).

**Why a validation set?** Decisions (choosing `C`, choosing the best epoch) are made on validation. The test set is touched once at the end, so its score stays honest.

---

### `src/analysis/side_effects.py`: the side-effect lexicon

- `SIDE_EFFECTS`: 22 side effects, each with the phrases that count as a mention (`"nausea"`, `"nauseous"`, `"nauseated"`).
- Regex with `\b` word boundaries: `"tired"` matches *I feel tired* but not *I recently retired*.
- `CONDITION_OVERLAP`: if the patient is being treated for depression, mentioning depression is the condition, not a side effect. Mental-health conditions are grouped because they overlap (a depression patient often writes about anxiety too).
- `detect_side_effects(text, condition)` returns a list like `["nausea", "headache"]`.

**Why a lexicon and not a model?** No labelled side-effect data exists here, and a lexicon is 100% explainable. Its limitation (it finds mentions, not causes) is stated in the README. Upgrading it to an NER model is listed as future work, which is a good thing to discuss in an interview.

---

### `src/analysis/eda.py`: step 3

- `plot_overview()`: rating distribution, review length, top conditions.
- `build_drug_insights()`: one row per drug with review count, average rating, % negative/neutral/positive, and the most common condition. Saved as `reports/insights/drug_insights.csv`.
- `build_side_effect_table()`: for each drug's **negative** reviews, counts each side effect and the share of negative reviews mentioning it.
- `plot_insights()`: overall side-effect chart, plus best and worst rated drugs with 300+ reviews (a minimum count, so one angry review can't make a drug "the worst").

The dashboard's Drug insights tab reads these CSVs. That's why they're committed to Git and the app works without re-running the pipeline.

---

### `src/models/evaluate.py`: shared scoring

Both models are scored by the same functions, so the comparison is fair.

- `compute_metrics()`: accuracy, **macro-F1**, weighted-F1, and F1 per class.
- `save_results()`: writes `reports/metrics/<model>.json`, a predictions CSV (used by error analysis), and a confusion matrix chart.
- The confusion matrix shows **row percentages**: each row is one true class, so the diagonal is that class's recall.

**Why macro-F1?** 66% of reviews are positive. A model that always answers "positive" gets 66% accuracy and is useless. Macro-F1 averages the F1 of each class equally, so the small neutral class counts as much as the big positive one.

---

### `src/models/baseline.py`: step 4

**TF-IDF** turns text into numbers:
- **TF** (term frequency): how often a word appears in this review. `sublinear_tf=True` uses 1 + log(count), so writing "great" five times doesn't count five times as much.
- **IDF** (inverse document frequency): rare words get high weight, common words like "the" get almost zero.
- `ngram_range=(1, 2)` adds word pairs, so `"not good"` and `"no side"` become features. This matters a lot for sentiment.
- `min_df=3` ignores words in fewer than 3 reviews (typos, names); `max_features=50000` keeps the vocabulary manageable.

**Logistic Regression** learns one weight per word per class. `class_weight="balanced"` makes mistakes on the rare neutral class cost more. `C` is the inverse of regularisation strength (small C = simpler model); it's chosen by trying 4 values on validation.

`top_words_per_class()` reads the learned weights and shows the most telling words per class. For neutral these are *however, but, yet, although*, which is a great finding to mention.

**Why have a baseline at all?** It's fast, interpretable and surprisingly strong. A transformer is only worth its cost if it beats this clearly.

---

### `src/models/transformer.py`: step 5 (the deep learning part)

**What is DistilBERT?** A transformer language model with 66M parameters. It was pre-trained on Wikipedia and books to predict hidden words, so it already understands English. "Distilled" means it was trained to copy the bigger BERT model: about 40% smaller and 60% faster, keeping about 97% of BERT's ability. **Fine-tuning** means adding a small classification layer on top and training the whole thing briefly on our labelled reviews.

**Why does it beat TF-IDF?** TF-IDF sees a bag of words with no order. DistilBERT uses **self-attention**: each word's representation depends on every other word in the sentence. So it can tell *"not bad at all"* from *"bad, not at all helpful"*, and understands that *"but"* flips the meaning of what came before.

Walk through the code in order:

1. **`ReviewDataset`**: a minimal PyTorch `Dataset`. `__len__` returns the count, `__getitem__` returns one (text, label) pair.
2. **`make_collate_fn()`**: tokenises a whole batch at once. `padding=True` pads only to the longest review in that batch (dynamic padding), not to 256, which saves a lot of compute. `truncation=True, max_length=256` cuts very long reviews.
3. **`DataLoader`**: batches and shuffles the training data each epoch.
4. **Model**: `AutoModelForSequenceClassification.from_pretrained(..., num_labels=3)` loads the pretrained body and adds a fresh 3-class head.
5. **Loss**: `CrossEntropyLoss` with **√-inverse-frequency class weights**. The neutral class is ~9% of data, so it gets a higher weight. Square root instead of full inverse, because full inverse makes the model over-predict neutral.
6. **Optimiser**: `AdamW` (Adam with decoupled weight decay). Bias and LayerNorm weights get no decay, which is standard practice.
7. **Schedule**: `get_linear_schedule_with_warmup`. The learning rate rises from 0 over the first 6% of steps, then falls linearly to 0. A warmup avoids large updates while the new classification head is still random, which could damage the pretrained weights.
8. **The training step** (the core of every PyTorch project):
   ```python
   logits = model(**batch).logits      # forward pass
   loss = loss_fn(logits, labels)      # how wrong were we?
   optimizer.zero_grad()               # clear old gradients
   loss.backward()                     # backward pass: compute gradients
   clip_grad_norm_(..., 1.0)           # cap gradient size (stability)
   optimizer.step()                    # update the weights
   scheduler.step()                    # update the learning rate
   ```
9. **Mixed precision** (`torch.autocast` + `GradScaler`, GPU only): most maths runs in 16-bit floats, about 2× faster with less memory. `GradScaler` scales the loss up before `backward()` so tiny gradients don't round to zero in 16-bit, then scales them back.
10. **`model.train()` vs `model.eval()`**: train mode turns dropout on (randomly drops neurons to reduce overfitting); eval mode turns it off for predictions. `@torch.no_grad()` in `predict_proba` skips gradient tracking, which is faster and uses less memory.
11. **Best checkpoint**: after each epoch it computes validation macro-F1 and saves the model only if it improved (`save_pretrained`). At the end it reloads the best checkpoint and evaluates on test once.

**Why a hand-written loop instead of Hugging Face `Trainer`?** `Trainer` hides all of the above. Writing it yourself shows you understand PyTorch, which the JD explicitly asks for, and makes every step explainable.

---

### `src/models/report.py`: step 6

Reads every `reports/metrics/*.json`, writes `model_comparison.csv`, draws the comparison chart, and **rewrites the README results table** between the `<!-- RESULTS_TABLE_START -->` and `<!-- RESULTS_TABLE_END -->` markers. The best model gets a ⭐. That's why the DistilBERT numbers appear automatically after you run the Kaggle notebook.

---

### `src/models/error_analysis.py`: step 7

Uses DistilBERT's predictions if available, otherwise the baseline's. Writes `reports/error_analysis.md` with:
1. The most common mistake types (e.g. true positive → predicted neutral)
2. How much of the error comes from borderline ratings (4–7)
3. Accuracy on reviews with contrast words (*but, however*) vs without
4. Accuracy by review length
5. The most confident mistakes, with the review text
6. Sample neutral reviews the model missed

**Why it matters:** anyone can report a score. Explaining where a model fails, and whether that's the model's fault or noisy labels, is what interviewers remember.

---

### `src/models/predict.py`: used by the app

- `SentimentPredictor` loads DistilBERT if `models/distilbert/` exists and PyTorch works; otherwise the baseline. If PyTorch is installed but broken, it falls back to the baseline instead of crashing.
- `explain()` uses the baseline's weights: each word's contribution = its TF-IDF value × (its weight for the predicted class − the average weight across classes). The app highlights the top contributors.
- `baseline_version_problem()` checks the scikit-learn version the model was saved with. Pickled models can silently mispredict on other versions (we found this exact bug in the readmission project).

---

### `app/streamlit_app.py`: the dashboard

- `@st.cache_resource` loads the model once; `@st.cache_data` loads the insight CSVs once.
- **Analyze a review**: example buttons write into `st.session_state["review_text"]`, which fills the text box. The `highlight()` function escapes the text with `html.escape` first (so a review can't inject HTML), then wraps the important words in `<mark>`.
- **Drug insights**: reads the CSVs from the EDA step and draws the drug's side-effect chart.
- **Model performance**: comparison table, charts, and the error report.

---

### `main.py`, `tests/`, CI, notebook

- `main.py`: runs steps in order. Imports happen inside each step, so the baseline works even if PyTorch isn't installed. The transformer step checks for a GPU and explains what to do if there isn't one.
- `tests/test_prepare.py`: cleaning, label boundaries (parametrised over 6 ratings), and a full `prepare_data()` run on fake data that checks duplicates and leaked test rows are removed. `monkeypatch` points the config paths at a temporary folder.
- `tests/test_side_effects.py`: multiple effects, whole-word matching, condition awareness.
- `tests/test_models.py`: the baseline learns obvious words; the **transformer test builds a tiny DistilBERT-shaped model locally** (32-dim, 1 layer, no download) and runs the real training loop end to end. It is skipped automatically if PyTorch isn't usable.
- `.github/workflows/tests.yml`: installs CPU-only PyTorch (small and fast) and runs the tests on every push.
- `notebooks/finetune_distilbert_gpu.ipynb`: clones your repo on Kaggle/Colab, runs the pipeline on a free GPU, and zips the results for you to download.

---

## 3. Running DistilBERT: step by step

1. Push the repo to GitHub first. The notebook clones it, so replace `<your-username>` in cell 1.
2. Kaggle → *New Notebook* → *File → Import Notebook* → upload `notebooks/finetune_distilbert_gpu.ipynb`.
3. Right panel: *Accelerator → GPU T4 x2*, and *Internet → On* (needs a phone-verified Kaggle account).
4. *Run All*. Cell 2 must print `GPU available: True`.
5. When it finishes, download `reports.zip` from the Output panel. Unzip it into your local project; it overwrites `reports/` and `README.md` with real DistilBERT numbers. Commit and push.
6. Update the README's "Reading the baseline" paragraph with one or two sentences on how DistilBERT compares, especially on the neutral class.
7. Optional: unzip `distilbert.zip` into `models/` to use DistilBERT in your local app.

If you're short on time, uncomment the `TRAIN_SUBSET = 40_000` cell for a run of roughly 15 minutes. Mention that in the README if you do.

---

## 4. How to experiment

| Try this | Where | What you'll learn |
|---|---|---|
| Remove the leakage fix (comment out the `overlap` lines) | `prepare.py` | the test score jumps; that's leakage in action |
| `TFIDF_NGRAMS = (1, 1)` | `config.py` | how much word pairs like "not good" help |
| Remove `class_weight="balanced"` | `baseline.py` | accuracy goes up, neutral F1 collapses: why macro-F1 matters |
| `MAX_LENGTH = 128` | `config.py` | faster training; does cutting long reviews hurt? |
| `EPOCHS = 3` | `config.py` | does it overfit? watch validation macro-F1 per epoch |
| Change `TRANSFORMER_MODEL_NAME` to `"microsoft/MiniLM-L12-H384-uncased"` | `config.py` | an even smaller model: the speed vs accuracy trade-off |

---

## 5. Interview questions you should be ready for

**Q: What's the most interesting thing you found?**
That 32,131 official test reviews were exact copies of training reviews, because Drugs.com lists one review under both the brand and generic names. I de-duplicated and removed all overlaps before evaluating, so my numbers are honest.

**Q: Why macro-F1 and not accuracy?**
66% of reviews are positive, so always predicting positive gives 66% accuracy. Macro-F1 weights each class equally, so the 9% neutral class matters as much as the positive one.

**Q: How did you handle class imbalance?**
Baseline: `class_weight="balanced"`. DistilBERT: cross-entropy weighted by the square root of inverse class frequency, which helps neutral without over-predicting it. Model selection by validation macro-F1.

**Q: Explain TF-IDF.**
Each review becomes a vector with one value per word: how often the word appears in this review, times how rare it is across all reviews. Rare, telling words get high weight; common words get almost none.

**Q: Why does DistilBERT beat TF-IDF?**
TF-IDF ignores word order. DistilBERT's self-attention builds each word's meaning from the whole sentence, so it understands negation and contrast ("works, but the side effects…"), which is exactly where the baseline fails.

**Q: What is fine-tuning? Why not train from scratch?**
Taking a model pre-trained on huge amounts of text and training it briefly on our task with a small learning rate. Training from scratch would need far more data and compute; the pre-trained model already knows English.

**Q: Why warmup?**
At the start the classification head is random, so gradients are large and noisy. A small learning rate at first prevents these early updates from damaging the pretrained weights.

**Q: What does mixed precision do?**
Runs most calculations in 16-bit floats on the GPU: about twice as fast and less memory. A GradScaler multiplies the loss before backward so small gradients don't underflow to zero.

**Q: What does gradient clipping do?**
Caps the overall size of the gradient (at 1.0 here) so one bad batch can't make a huge, destabilising update.

**Q: What's the difference between `model.train()` and `model.eval()`?**
Train mode enables dropout; eval mode disables it so predictions are deterministic. With `torch.no_grad()` we also skip gradient tracking during prediction.

**Q: What does dynamic padding mean?**
Each batch is padded only to its longest review rather than to a fixed 256 tokens, so short-review batches run much faster.

**Q: Why is the neutral class so hard?**
It's small, and neutral reviews mix praise and complaints. Also, ratings are noisy: 38% of the baseline's errors come from ratings 4–7, where a 6 and a 7 often read the same.

**Q: How does your side-effect extraction work, and what are its limits?**
A lexicon of 22 side effects with regex whole-word matching, applied to negative reviews. It's condition-aware: an antidepressant patient mentioning depression isn't counted. It finds mentions, not causes. The next step would be a token-classification (NER) model.

**Q: How would you deploy this?**
Push the fine-tuned model to the Hugging Face Hub, serve it with FastAPI in Docker (I built REST APIs in my internship), and add monitoring for drift as new drugs and slang appear.

**Q: How does this relate to Evernorth?**
Evernorth runs one of the largest pharmacy benefit businesses. Patient feedback about medicines can flag drugs with poor experience and side effects that drive patients to stop taking them, which affects adherence programmes and formulary decisions.

---

## 6. Limitations to state honestly

- Ratings are noisy labels.
- The lexicon finds mentions, not causes, and can't always handle negation ("no nausea at all").
- Reviewers are self-selected; ratings are polarised.
- Educational project: not medical advice.
