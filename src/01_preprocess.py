"""Faithful reproduction of the published preprocessing, with the protocol errors left OUT
(no fitting on the full dataset happens here -- only text cleaning, which is row-independent)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd
from textblob import TextBlob

from common import STOP, SYMPTOMS, clean_tweet

def build_social(path="data/ebola_tweets.csv"):
    df = pd.read_csv(path)
    n_raw = len(df)
    df = df.drop_duplicates(subset=["timestamp","user","text"])
    df = df.rename(columns={"text":"tweets"})
    cleaned = [clean_tweet(t) for t in df.tweets.tolist()]
    pol = [TextBlob(t).sentiment.polarity for t in cleaned]
    out = pd.DataFrame({"tweet": cleaned, "polarity": pol})
    out["sentiment_class"] = np.where(out.polarity > 0, 0, np.where(out.polarity < 0, 1, 2))
    low = out.tweet.str.lower()
    out["symptom_keyword_count"] = [sum(1 for k in SYMPTOMS if k in t) for t in low]
    out["label"] = (out.symptom_keyword_count >= 1).astype(int)
    out.attrs["n_raw"] = n_raw
    return out

def build_clinical(path="data/covid 19 clinical dataset.xlsx"):
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_excel(path).drop(columns=["Patient ID"])
    df = df.rename(columns={"SARS-Cov-2 exam result":"labels"})
    dead = [c for c in df.columns if df[c].replace(0, np.nan).isna().all()]
    df = df.drop(columns=dead)
    df["labels"] = df["labels"].map({"positive":1,"negative":0})
    cat = [c for c in df.select_dtypes(include=["object","category"]).columns]
    df[cat] = df[cat].fillna(0)
    for c in cat:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    y = df["labels"].values
    X = df.drop(columns=["labels"])
    return X, y, {"dropped_all_zero_or_nan": dead, "n_categorical": len(cat)}

if __name__ == "__main__":
    s = build_social()
    print("SOCIAL  raw=%d  dedup=%d  pos=%d (%.1f%%)  neg=%d" %
          (s.attrs["n_raw"], len(s), s.label.sum(), 100*s.label.mean(), (1-s.label).sum()))
    print("  sentiment classes:", s.sentiment_class.value_counts().to_dict(), " (0=pos,1=neg,2=neutral)")
    print("  empty cleaned tweets:", (s.tweet.str.len()==0).sum())
    X, y, meta = build_clinical()
    print("CLINICAL n=%d  predictors=%d  pos=%d (%.1f%%)" % (len(X), X.shape[1], y.sum(), 100*y.mean()))
    print("  dropped all-zero/all-NaN cols:", len(meta["dropped_all_zero_or_nan"]), meta["dropped_all_zero_or_nan"])
    print("  categorical encoded:", meta["n_categorical"])
    s.to_csv("social_prepped.csv", index=False)
    X.to_csv("clinical_X.csv", index=False); np.save("clinical_y.npy", y)
