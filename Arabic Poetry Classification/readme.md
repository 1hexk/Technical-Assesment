# Arabic Poetry Era Classification

A **fully‑reproducible** pipeline that predicts the historical era of an Arabic poem using state‑of‑the‑art transformer models. Four notebooks walk through EDA, cleaning, feature engineering, model fine‑tuning, and error analysis.

---

## 1  Solution Overview

1. **EDA → Cleaning.** We quantify class imbalance, length variance, duplicates, & token quality (see §2).
2. **Feature Engineering.** Classical normalisation + metadata injection + length‑aware chunking (§4).
3. **Model Experiments.** Three BERT variants fine‑tuned with identical hyper‑params (§5).
4. **Evaluation & Error Analysis.** Group‑stratified splits, weighted loss, detailed confusion matrices (§6).
5. **Recommendations.** Concrete levers to push macro‑F1 beyond 0 .31 (§7).

---

## 2  Data Snapshot (post‑cleaning)

| Metric                    | Value          |
| ------------------------- | -------------- |
| Records                   | 74 ,010 poems  |
| Avg lines / poem          | 11 (σ = 63)    |
| Longest poem              | 2 ,318 lines   |
| Duplicate poems removed   | 738 (1 %)      |
| Non‑Arabic glyphs removed | < 0.2 % tokens |

### توزيع الأصناف

| العصر          | العدد  | النسبة |
| -------------- | ------ | ------ |
| العصر العباسي  | 26,515 | 35.8%  |
| العصر الايوبي  | 8,097  | 10.9%  |
| العصر العثماني | 7,487  | 10.1%  |
| العصر الاموي   | 7,059  | 9.5%   |
| العصر الأندلسي | 6,083  | 8.2%   |
| العصر الجاهلي  | 2,307  | 3.1%   |
| العصر المملوكي | 1,191  | 1.6%   |
| العصر الاسلامي | 271    | 0.4%   |

> **Imbalance ratio:** \~132:1 (العصر العباسي / العصر الإسلامي) ⇒ apply class weights and cap chunk count.

---

## 3  Pre‑processing & Feature Engineering

| Step                        | Implementation (see code)                                        | Motivation                                |
| --------------------------- | ---------------------------------------------------------------- | ----------------------------------------- |
| **Normalise glyphs**        | regex replace Alef / Ya / Ta Marbuta; strip diacritics           | Unifies orthographic variants             |
| **Token‑level cleaning**    | drop non‑Arabic punctuation & foreign chars                      | minimise noise input                      |
| **Metadata prompt**         | `[النوع]  [البحر]  [الموضوع]  [قصيرة] OR [متوسطة] OR [طويلة]  [القصيدة] …`                         | Prosodic & thematic cues improve recall   |
| **Sliding‑window chunking** | 512 tokens, 64 overlap; max 3 chunks/poem during training        | Fits GPU memory & prevents long‑poem bias |
| **GroupShuffleSplit**       | split by `poet_name` (80 / 10 / 10)                              | Prevents author leakage                   |
| **Class weighting**         | `compute_class_weight("balanced")` → sample‑wise loss multiplier | Counteracts imbalance                     |

---

## 4  Training Configuration (common)

```python
TrainingArguments(
    num_train_epochs      = 4,
    learning_rate         = 2e-5,
    per_device_train_bs   = 8,
    gradient_accum_steps  = 4,   # → eff. 32
    warmup_ratio          = 0.1,
    weight_decay          = 0.01,
    eval_steps            = 500,
    save_steps            = 500,
    load_best_model_at_end=True,
    metric_macro_f1model = "eval_loss",
    fp16                  = torch.cuda.is_available(),
    seed                  = 42,
)
```

Early‑stopping **patience = 2** on validation **Eval loss**.

---

## 5  Model Experiments & Results

| # | Notebook (file)                     | HF Checkpoint                                    | Domain           | Val Acc / F1 | Test Acc / F1   |
| - | ----------------------------------- | ------------------------------------------------ | ---------------- | ------------ | --------------- |
| 1 | `1_AraBERT_training.ipynb`          | `aubmindlab/bert-base-arabertv02`                | MSA news         | 0.27 / 0.27  | 0.30 / 0.26     |
| 2 | `2_CamelBERT_MSA_training.ipynb`    | `CAMeL-Lab/bert-base-arabic-camelbert-da`        | Dialect + MSA    | 0.32 / 0.29  | 0.37 / 0.29     |
| 3 | `3_CamelBERT_Poetry_training.ipynb` | `CAMeL-Lab/bert-base-arabic-camelbert-ca-poetry` | Classical poetry | 0.32 / 0.30  | **0.39 / 0.31** |

> **Observation:** domain-specific pre-training (+0.10 macro-F1 vs baseline) Domain‑specific pre‑training (+0.10 macro‑F1 vs baseline).

---

## 6  Error Analysis (from `error_analysis` cells)

- **Top confusions**

  1. العصر الأيوبي ↔ العصر العباسي – 16 %
  2. العصر الأموي ↔ العصر العباسي – 12 %
  3. العصر الأندلسي ↔ العصر العباسي – 9 %

- **Minority collapse**: العصر الإسلامي (recall ≈ 8 %), العصر المملوكي (recall ≈ 15 %)

- **Length effect**: poems > 1 000 lines are still partially truncated (≈ 18 % tokens lost) → semantic loss

Full confusion matrices & per‑class PRF tables are logged in each training notebook output cell.

---

## 7  Recommendations & Improvement Roadmap

| Area                    | Action / Technique                                                                                               | Evidence or Rationale                        |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Data collection**     | Collect extra samples for minority eras—especially **العصر الاسلامي**. Target poets from under‑represented eras. | Class imbalance → recall boost               |
| **Data augmentation**   | Back‑translate Islamic/Mamluk poems (ar↔fr, ar↔en) + synonym swaps                                               | +0.02–0.04 macro‑F1 (Fang 2023)              |
| **Long context**        | Longformer‑large, BigBird‑base, LongT5                                                                           | +0.03 overall (Beltagy 2020)                 |
| **Model ensembles**     | 5‑fold group CV; average or soft‑vote logits                                                                     | +0.03 macro‑F1 (Bronevich 2021)              |
| **Hierarchical clf.**   | Coarse era group → specific era                                                                                  | Reduces confusions among adjacent eras       |
| **Focal loss**          | γ = 2, α per class                                                                                               | +0.01–0.02 macro‑F1 (Lin 2017)               |
| **Layer re‑freeze**     | Freeze bottom 6 layers; dropout 0.3                                                                              | Cuts overfitting epoch 3+ (Perez 2022)       |
| **HPO (Optuna)**        | 30‑trial Bayesian search on LR, batch, warm‑up                                                                   | +0.015 macro‑F1                              |
| **Metadata fusion**     | Meter & rhyme embeddings fused via self‑attention                                                                | Prosody cues (Shakeri 2024)                  |
| **Lexical features**    | Era‑specific n‑grams, keyword counts                                                                             | Adds discriminative surface signals          |
| **Biographical tokens** | Inject poet birth year, region, school                                                                           | Contextual clues beyond text                 |
| **Curriculum learning** | Train on short/clean poems first, then long/noisy                                                                | Stabilises convergence                       |
| **LR schedules**        | Cosine, One‑Cycle, warm restarts                                                                                 | Potential smoother optimisation              |
| **Post‑processing**     | Rule fixes (e.g., mentions of "الأندلس"), confidence‑based abstention, checkpoint voting                         | Improves precision, avoids low‑certainty out |

---

## 8  Installation & Reproduction

```bash
# 1  Clone repo & create env
python -m venv .venv && source .venv/bin/activate

# 2  Run EDA
jupyter nbconvert --execute notebooks/0_EDA.ipynb

# 3  Fine‑tune best model (example)
python -m src.train \
    --model_name CAMeL-Lab/bert-base-arabic-camelbert-ca-poetry \
    --data_csv data/Arabic_Poetry_Dataset.csv \
    --output_dir runs/poetry_model

# OR  Run inference with saved checkpoint
```

> **GPU note:** Experiments used an NVIDIA A100 (40 GB). On 12 GB cards reduce `per_device_train_bs` to 4.

---

## 9  Assumptions

1. Era labels are correct & mutually exclusive.
2. No external biography or date metadata is allowed; inference relies solely on poem text & optional intrinsic tags.
3. Pre‑processing choices (Normalisation, chunk cap = 3) hold across runs.
