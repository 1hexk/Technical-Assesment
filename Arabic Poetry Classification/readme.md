# Arabic Poetry Era Classification



A **fully‑reproducible** pipeline that predicts the historical era of an Arabic poem using state‑of‑the‑art transformer/traditional models. walk through EDA, cleaning, feature engineering, model fine‑tuning, and error analysis.

---

## 1  Solution Overview

1. **EDA → Cleaning.** We quantify class imbalance, length variance, duplicates, & token quality (see §2).
2. **Feature Engineering.** Classical normalisation + metadata injection + length‑aware chunking (§4).
3. **Model Experiments.** Six traditional classifiers and three BERT variants fine‑tuned with identical hyper‑params (§5).
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

### Label distributions 

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

| Step                        | Implementation (see code)                                                  | Motivation                                |
| --------------------------- | -------------------------------------------------------------------------- | ----------------------------------------- |
| **Normalise glyphs**        | regex replace Alef / Ya / Ta Marbuta; strip diacritics                     | Unifies orthographic variants             |
| **Token‑level cleaning**    | drop non‑Arabic punctuation & foreign chars                                | minimise noise input                      |
| **Metadata prompt**         | `[النوع]  [البحر]  [الموضوع]  [قصيرة] OR [متوسطة] OR [طويلة]  [القصيدة] …` | Prosodic & thematic cues improve recall   |
| **Sliding‑window chunking** | 512 tokens, 64 overlap; max 3 chunks/poem during training                  | Fits GPU memory & prevents long‑poem bias |
| **GroupShuffleSplit**       | split by `poet_name` (80 / 10 / 10)                                        | Prevents author leakage                   |
| **Class weighting**         | `compute_class_weight("balanced")` → sample‑wise loss multiplier           | Counteracts imbalance                     |

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

| # | Model / Notebook              | Checkpoint or Sklearn Estimator                  | Domain / Type           | Test Acc | Macro F1 |
| - | ----------------------------- | ------------------------------------------------ | ----------------------- | -------- | -------- |
| 1 | `AraBERT_training`            | `aubmindlab/bert-base-arabertv02`                | MSA news (BERT)         | 0.30     | 0.26     |
| 2 | `CamelBERT_MSA_training`      | `CAMeL-Lab/bert-base-arabic-camelbert-da`        | Dialect + MSA (BERT)    | 0.37     | 0.29     |
| 3 | `CamelBERT_Poetry_trainingyn` | `CAMeL-Lab/bert-base-arabic-camelbert-ca-poetry` | Classical poetry (BERT) | **0.39** | **0.31** |
| 4 | `scikit_knn`                  | K‑Nearest Neighbors (k=5)                        | TF‑IDF + KNN            | 0.32     | 0.20     |
| 5 | `scikit_svm_linear`           | Linear SVM (C=1.0)                               | TF‑IDF + Linear SVM     | 0.36     | 0.06     |
| 6 | `scikit_svm_rbf`              | RBF SVM (C=10, γ=1e‑3)                           | TF‑IDF + RBF SVM        | 0.45     | 0.22     |
| 7 | `scikit_decision_tree`        | Decision Tree (depth=none)                       | Bag‑of‑Words + DT       | 0.38     | 0.11     |
| 8 | `scikit_random_forest`        | Random Forest (n=200)                            | Bag‑of‑Words + RF       | 0.36     | 0.06     |
| 9 | `scikit_adaboost`             | AdaBoost (n=200)                                 | Bag‑of‑Words + AB       | 0.36     | 0.13     |

> **Observation:** Transformer models substantially outperform traditional ML baselines (+7–13 pp accuracy), but RBF‑SVM remains a competitive non‑deep baseline when tuned.

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

## 7  Installation & Reproduction

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

## 8  Assumptions

1. Era labels are correct & mutually exclusive.
2. No external biography or date metadata is allowed; inference relies solely on poem text & optional intrinsic tags.
3. Pre‑processing choices (Normalisation, chunk cap = 3) hold across runs.

---

## 9  Conclusion

After **six traditional classifiers**, **three fine‑tuned transformers**, and an arsenal of techniques—upsampling, class‑weighted losses, focal loss, extensive feature engineering, and hyper‑parameter tuning—the best macro‑F1 hovers around **0.31**. In plain terms: **the models are still struggling**.

What this tells us:

1. **Signal‑to‑noise** is low. Linguistic overlap between eras dilutes the discriminative power of surface text alone.
2. **Label quality** may be shaky. Historical era tags can be ambiguous or inconsistently annotated.
3. **Data scarcity** for minority eras (e.g., العصر الإسلامي) is too extreme for deep models to generalise.

> **Implication**: Further tweaks to architectures are unlikely to move the needle. Real gains will come from **data‑centric work**—curating more balanced, high‑fidelity samples and injecting external knowledge (author metadata, historical context) rather than just throwing more models at the problem.

---

