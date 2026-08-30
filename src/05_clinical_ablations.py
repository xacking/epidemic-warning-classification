import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import wilcoxon
import tensorflow as tf, os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from tensorflow.keras.layers import Input,Conv1D,GlobalMaxPooling1D,Bidirectional,LSTM,Dense,Concatenate,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
tf.get_logger().setLevel("ERROR"); SEED=0

raw=pd.read_excel("data/covid 19 clinical dataset.xlsx").drop(columns=["Patient ID"]).rename(columns={"SARS-Cov-2 exam result":"labels"})
raw["labels"]=raw["labels"].map({"positive":1,"negative":0})
num=raw.select_dtypes(include=[np.number]).drop(columns=["labels"],errors="ignore")
keep=[c for c in num.columns if num[c].notna().mean()>=0.10]
sub=raw[keep+["labels"]].dropna().reset_index(drop=True)
X=sub[keep].copy(); y=sub.labels.values.astype(int)
def col(sfx): return [c for c in keep if sfx.lower() in c.lower().replace("\xa0"," ")][0]
LEU,LYM,MON,NEU_ish,PLT,EOS = col("Leukocytes"),col("Lymphocytes"),col("Monocytes"),None,col("Platelets"),col("Eosinophils")

def ratios(df):
    d=df.copy(); e=1e-6
    d["ratio_leu_lym"]=d[LEU]-d[LYM]          # values are z-scored, so differences are log-ratio-like
    d["ratio_plt_lym"]=d[PLT]-d[LYM]
    d["ratio_leu_plt"]=d[LEU]-d[PLT]
    d["ratio_eos_leu"]=d[EOS]-d[LEU]
    d["lym_x_leu"]=d[LYM]*d[LEU]
    return d

def sens_at_spec(yt,p,t):
    thr=np.quantile(p[yt==0],t); return ((p>=thr)&(yt==1)).sum()/max((yt==1).sum(),1)

def evaluate(name, make_X, model="rf", folds=5, reps=2, collect=False):
    cv=RepeatedStratifiedKFold(n_splits=folds,n_repeats=reps,random_state=SEED)
    au,ap,s90,s80,briers=[],[],[],[],[]
    per=[]
    for fold,(tr,te) in enumerate(cv.split(X,y)):
        A,B=make_X(tr,te,fold)
        if model=="rf":
            m=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=2,class_weight="balanced_subsample").fit(A,y[tr])
        elif model=="rf_cal":
            base=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=2,class_weight="balanced_subsample")
            m=CalibratedClassifierCV(base,method="isotonic",cv=3).fit(A,y[tr])
        p=m.predict_proba(B)[:,1]
        a=roc_auc_score(y[te],p); au.append(a); ap.append(average_precision_score(y[te],p))
        s90.append(sens_at_spec(y[te],p,.90)); s80.append(sens_at_spec(y[te],p,.80))
        briers.append(np.mean((p-y[te])**2)); per.append(a)
    print(f"{name[:56]:58s} AUROC {np.mean(au):.3f}±{np.std(au,ddof=1):.3f}  AP {np.mean(ap):.3f}  "
          f"sens@sp90 {np.mean(s90):.3f}  sens@sp80 {np.mean(s80):.3f}  Brier {np.mean(briers):.3f}",flush=True)
    return dict(name=name,auroc=np.mean(au),auroc_sd=np.std(au,ddof=1),ap=np.mean(ap),
                s90=np.mean(s90),s80=np.mean(s80),brier=np.mean(briers),per_fold=per)

Xv=X.values.astype(float); Xr=ratios(X).values.astype(float)
def plain(tr,te,f): return Xv[tr],Xv[te]
def withratios(tr,te,f): return Xr[tr],Xr[te]

def deep(tr,te,fold):
    tf.keras.utils.set_random_seed(SEED+fold)
    mm=MinMaxScaler().fit(Xv[tr]); A,B=mm.transform(Xv[tr]),mm.transform(Xv[te])
    inp=Input(shape=(A.shape[1],1))
    c=GlobalMaxPooling1D()(Conv1D(64,3,activation="relu")(Conv1D(32,3,activation="relu")(inp)))
    feat=Concatenate()([c,Bidirectional(LSTM(32))(inp)])
    clf=Model(inp,Dense(1,activation="sigmoid")(Dense(32,activation="relu")(Dropout(.3)(feat))))
    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
    xa,xv2,ya,yv=train_test_split(A[...,None],y[tr],test_size=.15,random_state=SEED,stratify=y[tr])
    cw={0:1.,1:float((y[tr]==0).sum()/(y[tr]==1).sum())}
    clf.fit(xa,ya,validation_data=(xv2,yv),epochs=40,batch_size=32,verbose=0,class_weight=cw,
            callbacks=[EarlyStopping(patience=6,restore_best_weights=True)])
    ext=Model(inp,feat)
    Ftr=np.hstack([ext.predict(A[...,None],verbose=0),A]); Fte=np.hstack([ext.predict(B[...,None],verbose=0),B])
    sc=StandardScaler().fit(Ftr); return sc.transform(Ftr),sc.transform(Fte)

print("=== Complete-case cohort (n=598, 81 positive), 5-fold x 2 ===")
r=[]
r.append(evaluate("D1  RF, 500 trees, raw features","",))if False else None
r.append(evaluate("D1  RF, 500 trees (recommended)",plain))
r.append(evaluate("D2  RF + engineered cell ratios",withratios))
r.append(evaluate("D3  RF, isotonic-calibrated",plain,model="rf_cal"))
r.append(evaluate("D4  CNN+BiLSTM(trained)+clinical+RF",deep))
r=[x for x in r if x]
base=[x for x in r if x["name"].startswith("D1")][0]
print("\nPaired Wilcoxon on AUROC vs D1:")
for x in r:
    if x is base: continue
    a=np.array(base["per_fold"]); b=np.array(x["per_fold"])
    p=wilcoxon(a,b).pvalue if not np.allclose(a,b) else float("nan")
    print(f"  {x['name'][:52]:54s} {x['auroc']-base['auroc']:+.4f}  p={p:.4f}")
pd.DataFrame([{k:v for k,v in x.items() if k!='per_fold'} for x in r]).to_csv("results/clinical/IMPROVE2_results.csv",index=False)
print("\nsaved IMPROVE2_results.csv")
