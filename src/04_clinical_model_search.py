"""Can the clinical task be improved? Test concrete hypotheses under the same protocol."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
try:
    from xgboost import XGBClassifier; HAS_XGB=True
except Exception: HAS_XGB=False
SEED=0

raw = pd.read_excel("data/covid 19 clinical dataset.xlsx").drop(columns=["Patient ID"]).rename(columns={"SARS-Cov-2 exam result":"labels"})
raw["labels"] = raw["labels"].map({"positive":1,"negative":0})
y = raw["labels"].values.astype(int)
num = raw.select_dtypes(include=[np.number]).drop(columns=["labels"], errors="ignore")
# keep numeric predictors with any data at all
num = num[[c for c in num.columns if num[c].notna().any()]]
MISS = num.isna().astype(int)                       # which assays were ordered
NTEST = MISS.shape[1] - MISS.sum(1)                 # how many assays were ordered
print(f"full cohort n={len(num)}  numeric predictors={num.shape[1]}  positives={y.sum()} ({y.mean():.1%})")
print(f"tests ordered per patient: median {int(NTEST.median())}, IQR {int(NTEST.quantile(.25))}-{int(NTEST.quantile(.75))}")
print(f"correlation between #tests ordered and positivity: {np.corrcoef(NTEST, y)[0,1]:+.3f}\n")

def sens_at_spec(yt, p, target):
    """sensitivity at a fixed specificity, the way a screening tool is specified"""
    thr = np.quantile(p[yt==0], target)
    return ((p >= thr) & (yt==1)).sum() / max((yt==1).sum(), 1)

def run(name, Xd, yd, model_fn, impute=True, folds=5, repeats=2):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    au, ap, s90, s80, sens5 = [], [], [], [], []
    for tr, te in cv.split(Xd, yd):
        A, B = Xd[tr], Xd[te]
        if impute:
            im = SimpleImputer(strategy="median").fit(A); A, B = im.transform(A), im.transform(B)
        m = model_fn().fit(A, yd[tr])
        p = m.predict_proba(B)[:, 1]
        au.append(roc_auc_score(yd[te], p)); ap.append(average_precision_score(yd[te], p))
        s90.append(sens_at_spec(yd[te], p, .90)); s80.append(sens_at_spec(yd[te], p, .80))
        tn, fp, fn, tp = confusion_matrix(yd[te], (p>=.5).astype(int), labels=[0,1]).ravel()
        sens5.append(tp/max(tp+fn,1))
    r = dict(model=name, n=len(yd), auroc=np.mean(au), auroc_sd=np.std(au, ddof=1), ap=np.mean(ap),
             sens_at_spec90=np.mean(s90), sens_at_spec80=np.mean(s80), sens_at_05=np.mean(sens5))
    print(f"{name[:58]:60s} AUROC {r['auroc']:.3f}±{r['auroc_sd']:.3f}  AP {r['ap']:.3f}  "
          f"sens@spec90 {r['sens_at_spec90']:.3f}  sens@spec80 {r['sens_at_spec80']:.3f}", flush=True)
    return r

RF   = lambda: RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=2, class_weight="balanced_subsample")
HGB  = lambda: HistGradientBoostingClassifier(random_state=42, max_iter=300, learning_rate=.06,
                                              max_leaf_nodes=15, l2_regularization=1.0,
                                              class_weight="balanced", early_stopping=True, validation_fraction=.15)
XGB  = lambda: XGBClassifier(n_estimators=400, learning_rate=.05, max_depth=4, subsample=.8,
                             colsample_bytree=.8, reg_lambda=2.0, eval_metric="logloss",
                             scale_pos_weight=float((y==0).sum()/(y==1).sum()), random_state=42, n_jobs=2)
LR   = lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3)

rows=[]
print("=== A. FULL COHORT (n=5,644) — no complete-case selection ===")
V = num.values
rows.append(run("A1  RF + median imputation (current baseline)", V, y, RF))
rows.append(run("A2  RF + imputation + missingness indicators", np.hstack([V, MISS.values]), y, RF))
rows.append(run("A3  RF + imputation + indicators + test count", np.hstack([V, MISS.values, NTEST.values.reshape(-1,1)]), y, RF))
rows.append(run("A4  HistGradientBoosting, native NaN handling", V, y, HGB, impute=False))
rows.append(run("A5  HistGradientBoosting + indicators", np.hstack([V, MISS.values]), y, HGB, impute=False))
rows.append(run("A6  HistGradientBoosting + indicators + count", np.hstack([V, MISS.values, NTEST.values.reshape(-1,1)]), y, HGB, impute=False))
if HAS_XGB:
    rows.append(run("A7  XGBoost, native NaN handling", V, y, XGB, impute=False))
    rows.append(run("A8  XGBoost + indicators + count", np.hstack([V, MISS.values, NTEST.values.reshape(-1,1)]), y, XGB, impute=False))
rows.append(run("A9  Test count alone (how many assays ordered)", NTEST.values.reshape(-1,1).astype(float), y, RF))

print("\n=== B. COMPLETE-CASE COHORT (n=598) — better model class ===")
keep=[c for c in num.columns if num[c].notna().mean()>=0.10]
sub=raw[keep+["labels"]].dropna(); Xc=sub[keep].values.astype(float); yc=sub.labels.values.astype(int)
rows.append(run("B1  RF (current, 500 trees)", Xc, yc, RF))
rows.append(run("B2  HistGradientBoosting", Xc, yc, HGB, impute=False))
if HAS_XGB: rows.append(run("B3  XGBoost", Xc, yc, XGB, impute=False))
rows.append(run("B4  L2 logistic regression", StandardScaler().fit_transform(Xc), yc, LR))

print("\n=== C. COHORT SPECTRUM — completeness threshold sweep (HGB + indicators) ===")
spec=[]
for t in (0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.30):
    kk=[c for c in num.columns if num[c].notna().mean()>=t]
    ss=raw[kk+["labels"]].dropna()
    if len(ss)<150 or ss.labels.sum()<20: 
        print(f"  thresh {t:.2f}: {len(kk):3d} features -> n={len(ss):5d}  (too small, skipped)"); continue
    Xs=ss[kk].values.astype(float); ys=ss.labels.values.astype(int)
    r=run(f"C   completeness>={t:.2f}: {len(kk)} feats, n={len(ss)}, pos={ys.sum()}", Xs, ys, HGB, impute=False)
    r["threshold"]=t; r["n_features"]=len(kk); r["prevalence"]=float(ys.mean()); spec.append(r)

pd.DataFrame(rows).to_csv("results/clinical/IMPROVE_results.csv", index=False)
pd.DataFrame(spec).to_csv("results/clinical/IMPROVE_cohort_spectrum.csv", index=False)
print("\nsaved IMPROVE_results.csv, IMPROVE_cohort_spectrum.csv")
