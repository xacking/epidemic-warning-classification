# Epidemic warning-signal and disease-status classification

Code and derived results for:

> F. O. Muhammed, M. A. Suleiman, S. E. Abdullahi & A. O. Ogar,
> **"An Explainable CNN–BiLSTM–Random Forest Framework for Epidemic
> Warning-Signal and Disease-Status Classification"**,
> submitted to the *Journal of the Nigerian Society of Physical Sciences*.

A convolutional and bidirectional-recurrent feature extractor feeding a Random
Forest classifier, applied under one architectural template to two independent
tasks:

- **Social media** — labelling Ebola-related tweets as symptom-bearing warning
  signals, using the 2023 Ugandan-outbreak corpus of Mirugwe *et al.*
- **Clinical** — labelling structured COVID-19 patient records as positive or
  negative for SARS-CoV-2, from routine complete-blood-count variables.

The two tasks use disjoint data on different diseases and are never fused. No
forecasting horizon is modelled: both are classification of evidence already
contained in the record.

## Protocol

Everything is evaluated by **five-fold stratified cross-validation repeated
twice** (ten folds). The tokenizer, imputer, scalers, classifiers and decision
thresholds are all fitted **inside the training partition of each fold**. No
statistic is estimated over the full dataset. Reported `±` values are the
standard deviation across folds; comparisons use the Wilcoxon signed-rank test
on paired per-fold scores.

## Headline results

### Social media task (n = 8,395; 1,289 positive, 15.4%)

| Configuration | Sens. | Prec. | F1 | AUROC |
|---|---|---|---|---|
| CNN-RF, V=100, untrained *(as originally implemented)* | 0.651 | 0.933 | 0.766 | 0.926 |
| CNN-BiLSTM-RF + sentiment, V=100, untrained | 0.697 | 0.935 | 0.798 | 0.938 |
| CNN-BiLSTM-RF + sentiment, V=100, **trained** | 0.767 | 0.934 | 0.842 | 0.940 |
| CNN-RF, V=5000, trained | 0.919 | 0.980 | **0.948** | 0.981 |
| CNN-BiLSTM-RF + sentiment, V=5000, trained | 0.918 | 0.975 | 0.946 | 0.984 |
| TF-IDF + logistic regression *(reference)* | 0.876 | 0.972 | 0.922 | 0.981 |

Two implementation corrections account for the entire gain, and both are
reported as ablations rather than assumed:

1. **The tokenizer vocabulary.** `num_words=100` maps most of the 22
   symptom-dictionary terms that define the label to a single shared
   out-of-vocabulary index. Raising it to 5,000 moves F1 from 0.842 to 0.946.
2. **Training the extractors.** Worth +0.044 F1 at the published vocabulary
   (p = 0.002).

Once both are fixed, **the recurrent branch and the sentiment feature
contribute nothing measurable**: CNN-RF alone scores F1 0.948 against the full
ensemble's 0.946 (p = 0.43), sentiment p = 0.28.

### Clinical task, complete-case cohort (n = 598; 81 positive, 13.5%)

| Configuration | AUROC | Sens. @ 0.5 | Sens. @ 90% spec. | Sens. @ 80% spec. |
|---|---|---|---|---|
| Random Forest on 18 laboratory variables | **0.893** | 0.310 | **0.711** | **0.803** |
| Random Forest, 50 trees | 0.887 | 0.347 | 0.667 | 0.797 |
| Histogram gradient boosting | 0.863 | 0.636 | 0.587 | 0.710 |
| CNN-BiLSTM-RF, trained | 0.875 | 0.383 | 0.661 | 0.802 |

No configuration improves on a Random Forest fitted directly to the laboratory
variables. What changes the clinical result is the **operating point**, not the
model: the same classifier recovers 31.0% of cases at a threshold of 0.5 and
71.1% at 90% specificity.

**Thresholds must come from out-of-bag predictions.** A threshold read off the
model's own training scores does not transfer — a Random Forest fits its
training data almost perfectly, so the in-sample 90th percentile of negative
scores delivers 66% specificity on held-out data instead of 90%. Out-of-bag
thresholds transfer cleanly (realised specificity 0.896 against a 0.90 target).

### Clinical task, full cohort (n = 5,644)

| Configuration | AUROC | AP | Sens. @ 90% spec. |
|---|---|---|---|
| Random Forest + median imputation | 0.617 | 0.143 | 0.115 |
| Histogram gradient boosting, native missing-value handling | **0.661** | **0.212** | **0.180** |
| *Negative control:* number of assays ordered, alone | 0.510 | 0.111 | 0.073 |

Two things follow. Imputing a laboratory value that was never ordered destroys
information a model able to represent absence can use. And the negative control
answers the obvious objection to the complete-case analysis: the ordering
decision alone carries **no** predictive information, so the complete-case
result is not an artefact of which patients a clinician chose to test.

The cohort structure is bimodal — there is no intermediate cohort. At a 6–10%
completeness threshold the data holds 18–24 variables for 242–598 patients
(AUROC 0.86–0.87); at 15% and above only four variables survive but all 5,644
records qualify (AUROC 0.63).

### Explainability

SHAP attributions over the fitted forest rank **leukocytes** (0.0334) and
**platelets** (0.0130) highest; patient age quantile ranks sixth (0.0029). The
named clinical variables account for **14.1%** of the model's total mean
absolute attribution — the remainder sits in learned features with no clinical
reading.

## Layout

```
src/        numbered scripts, run in order
results/    fold-level metrics, confusion matrices, SHAP rankings, figures
data/       access instructions only - no raw data (see data/README.md)
```

| Script | Produces |
|---|---|
| `01_preprocess.py` | cleaned corpora: `social_prepped.csv`, `clinical_X.csv`, `clinical_y.npy` |
| `02_social_experiments.py` | `results/social/FINAL_social_*` |
| `03_clinical_experiments.py` | `results/clinical/FINAL2_*` (operating points, OOB thresholds) |
| `04_clinical_model_search.py` | `results/clinical/IMPROVE_*` (model class, cohort spectrum) |
| `05_clinical_ablations.py` | `results/clinical/IMPROVE2_*` (ratios, calibration, deep ensemble) |
| `06_shap_analysis.py` | SHAP ranking, permutation importance, SHAP figures |
| `07_error_analysis.py` | `results/social/FINAL_error_analysis.json` |
| `08`–`11` (figure scripts) | `results/figures/*.png` |

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the two source files as described in data/README.md
cd src
python 01_preprocess.py          # ~4 min
python 02_social_experiments.py  # ~20 min on 2 CPU cores
python 03_clinical_experiments.py
python 04_clinical_model_search.py
python 05_clinical_ablations.py
python 06_shap_analysis.py
python 07_error_analysis.py
for f in 08_figures_main 09_figures_clinical 10_figure_architecture 11_figures_workflows; do python $f.py; done
```

All seeds are fixed (`random_state=0` for splits, `42` for forests). The scripts
expect the data files in the working directory; adjust the paths at the top of
`01_preprocess.py` if you keep them in `data/`.

### Note on NLTK

`01_preprocess.py` inlines NLTK's English stopword list rather than downloading
it, so the pipeline runs without network access. The list is identical to
`nltk.corpus.stopwords.words('english')`.

## A caution on the labels

On the social media task the class label is a **deterministic function of the
text**: a tweet is positive exactly when one of 22 dictionary terms appears in
it. The task is therefore the recovery of a keyword rule, and the high absolute
scores should be read in that light rather than as evidence of epidemiological
validity. This is stated in the paper's Limitations and repeated here because
anyone reusing this code should know it.

## License

Code: [MIT](LICENSE). Derived results in `results/`: CC-BY-4.0.
