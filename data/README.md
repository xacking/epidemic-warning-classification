# Data access

Raw data is **not** redistributed here. Both datasets are obtainable from their
original sources. Place the two files in this directory and the scripts in
`../src/` will find them.

## 1. Ebola tweet corpus — `ebola_tweets.csv`

Released with:

> A. Mirugwe, C. Tumuhimbise & J. Ashaba, "Sentiment Analysis of Social Media
> Data on Ebola Outbreak Using Deep Learning Classifiers", *Life* **14** (2024)
> 708. https://doi.org/10.3390/life14060708

Expected columns: `timestamp`, `user`, `text`, `favorite_count`,
`retweet_count`, `location`. 13,629 rows as distributed; 8,395 remain after
exact-duplicate removal on (`timestamp`, `user`, `text`), which is the working
corpus for every result reported.

Under the platform's terms only tweet IDs and a hydration route may be
redistributed, not raw tweet text.

**Note on provenance.** The timestamps in the distributed file run from
2023-01-03 to 2023-01-19. Earlier descriptions of this corpus, including in a
previous version of our own manuscript, gave a collection window of
20 September – 30 November 2022; that window describes the Ugandan outbreak,
not the collection period of this file.

## 2. COVID-19 clinical records — `covid 19 clinical dataset.xlsx`

Anonymised records from Hospital Israelita Albert Einstein, São Paulo, publicly
released on Kaggle and used by:

> S. Melchane, Y. Elmir & F. Kacimi, "COVID-19 detection from clinical and
> laboratory data using machine learning", *Procedia Computer Science* **239**
> (2024) 675. https://doi.org/10.1016/j.procs.2024.06.223

5,644 rows and 111 columns: a patient identifier, the target (`SARS-Cov-2 exam
result`), and 109 candidate predictors. Seven of the 109 are uniformly empty or
uniformly zero and are dropped by `src/01_preprocess.py`, leaving the 102
predictors used throughout. Laboratory values are released already standardised to
zero mean and unit variance. Cite the original source rather than re-hosting.

### Two cohorts

Both are constructed by `src/01_preprocess.py` and the experiment scripts:

| Cohort | Records | Predictors | Positive | Prevalence |
|---|---|---|---|---|
| Full, median-imputed | 5,644 | 102 (67 numeric, 35 categorical) | 558 | 9.9% |
| Complete-case | 598 | 18 | 81 | 13.5% |

The complete-case cohort keeps variables recorded for at least 10% of patients,
then keeps only records complete on all of them.
