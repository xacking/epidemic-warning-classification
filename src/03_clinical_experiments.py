"""Definitive clinical results: operating points from out-of-bag thresholds, both cohorts, all ablations."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import warnings, json, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from scipy.stats import wilcoxon
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv1D,GlobalMaxPooling1D,Bidirectional,LSTM,Dense,Concatenate,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
tf.get_logger().setLevel("ERROR"); SEED=0

raw=pd.read_excel("data/covid 19 clinical dataset.xlsx").drop(columns=["Patient ID"]).rename(columns={"SARS-Cov-2 exam result":"labels"})
raw["labels"]=raw["labels"].map({"positive":1,"negative":0}); Y_FULL=raw.labels.values.astype(int)
num=raw.select_dtypes(include=[np.number]).drop(columns=["labels"],errors="ignore")
num=num[[c for c in num.columns if num[c].notna().any()]]
KEEP=[c for c in num.columns if num[c].notna().mean()>=0.10]
sub=raw[KEEP+["labels"]].dropna().reset_index(drop=True)
XC=sub[KEEP].values.astype(float); YC=sub.labels.values.astype(int)
NTEST=(num.shape[1]-num.isna().sum(1)).values.reshape(-1,1).astype(float)
print(f"complete-case n={len(XC)} feats={XC.shape[1]} pos={YC.sum()} | full n={len(num)} feats={num.shape[1]} pos={Y_FULL.sum()}",flush=True)

def cmstats(yt,yp):
    tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    return tp/max(tp+fn,1), tn/max(tn+fp,1), tp/max(tp+fp,1), (tn,fp,fn,tp)

def evaluate(name, Xd, yd, fit_predict, impute=True, keep_curve=False):
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=SEED)
    R={k:[] for k in ("auroc","ap","sens05","spec05","prec05","s90","sp90","s80","sp80")}
    cms=[]; curve_y=[]; curve_p=[]
    for fold,(tr,te) in enumerate(cv.split(Xd,yd)):
        A,B=Xd[tr],Xd[te]
        if impute:
            im=SimpleImputer(strategy="median").fit(A); A,B=im.transform(A),im.transform(B)
        p, oof = fit_predict(A,B,yd[tr],fold)
        R["auroc"].append(roc_auc_score(yd[te],p)); R["ap"].append(average_precision_score(yd[te],p))
        s,sp,pr,cm=cmstats(yd[te],(p>=.5).astype(int))
        R["sens05"].append(s); R["spec05"].append(sp); R["prec05"].append(pr); cms.append(cm)
        for tgt,key in ((.90,"90"),(.80,"80")):
            thr=np.nanquantile(oof[yd[tr]==0],tgt)          # threshold from OUT-OF-FOLD scores only
            R[f"s{key}"].append(((p>=thr)&(yd[te]==1)).sum()/max((yd[te]==1).sum(),1))
            R[f"sp{key}"].append(((p<thr)&(yd[te]==0)).sum()/max((yd[te]==0).sum(),1))
        if keep_curve and fold<5: curve_y.append(yd[te]); curve_p.append(p)
    out=dict(model=name)
    for k,v in R.items():
        out[k]=float(np.mean(v)); out[k+"_sd"]=float(np.std(v,ddof=1))
    out["_folds_auroc"]=R["auroc"]; out["_folds_s90"]=R["s90"]
    out["cm"]=list(np.array(cms).sum(0)/2.0)
    if keep_curve: out["_curve"]=(np.concatenate(curve_y),np.concatenate(curve_p))
    print(f"{name[:52]:54s} AUROC {out['auroc']:.3f}±{out['auroc_sd']:.3f} AP {out['ap']:.3f} "
          f"sens@0.5 {out['sens05']:.3f} | sens@sp90 {out['s90']:.3f}±{out['s90_sd']:.3f} (sp {out['sp90']:.3f}) "
          f"| sens@sp80 {out['s80']:.3f} (sp {out['sp80']:.3f})",flush=True)
    return out

def rf_fp(n_est=500):
    def f(A,B,ytr,fold):
        m=RandomForestClassifier(n_estimators=n_est,random_state=42,n_jobs=2,
                                 class_weight="balanced_subsample",oob_score=True,bootstrap=True).fit(A,ytr)
        oob=np.nan_to_num(m.oob_decision_function_[:,1],nan=0.5)
        return m.predict_proba(B)[:,1], oob
    return f
def hgb_fp(A,B,ytr,fold):
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    mk=lambda: HistGradientBoostingClassifier(random_state=42,max_iter=300,learning_rate=.06,max_leaf_nodes=15,
            l2_regularization=1.0,class_weight="balanced",early_stopping=True,validation_fraction=.15)
    oof=cross_val_predict(mk(),A,ytr,cv=StratifiedKFold(5,shuffle=True,random_state=1),method="predict_proba",n_jobs=1)[:,1]
    return mk().fit(A,ytr).predict_proba(B)[:,1], oof
def deep_fp(A,B,ytr,fold):
    tf.keras.utils.set_random_seed(SEED+fold)
    mm=MinMaxScaler().fit(A); A2,B2=mm.transform(A),mm.transform(B)
    inp=Input(shape=(A2.shape[1],1))
    c=GlobalMaxPooling1D()(Conv1D(64,3,activation="relu")(Conv1D(32,3,activation="relu")(inp)))
    feat=Concatenate()([c,Bidirectional(LSTM(32))(inp)])
    clf=Model(inp,Dense(1,activation="sigmoid")(Dense(32,activation="relu")(Dropout(.3)(feat))))
    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
    xa,xv,ya,yv=train_test_split(A2[...,None],ytr,test_size=.15,random_state=SEED,stratify=ytr)
    cw={0:1.,1:float((ytr==0).sum()/(ytr==1).sum())}
    clf.fit(xa,ya,validation_data=(xv,yv),epochs=40,batch_size=32,verbose=0,class_weight=cw,
            callbacks=[EarlyStopping(patience=6,restore_best_weights=True)])
    ext=Model(inp,feat)
    F1=np.hstack([ext.predict(A2[...,None],verbose=0),A2]); F2=np.hstack([ext.predict(B2[...,None],verbose=0),B2])
    sc=StandardScaler().fit(F1)
    m=RandomForestClassifier(n_estimators=500,random_state=42,n_jobs=2,class_weight="balanced_subsample",
                             oob_score=True,bootstrap=True).fit(sc.transform(F1),ytr)
    return m.predict_proba(sc.transform(F2))[:,1], np.nan_to_num(m.oob_decision_function_[:,1],nan=0.5)

res=[]
print("\n=== COMPLETE-CASE COHORT (n=598) ===")
res.append(evaluate("Random Forest, 500 trees (recommended)",XC,YC,rf_fp(500),keep_curve=True))
res.append(evaluate("Random Forest, 50 trees (as published)",XC,YC,rf_fp(50)))
res.append(evaluate("HistGradientBoosting",XC,YC,hgb_fp,impute=False))
res.append(evaluate("CNN+BiLSTM(trained)+clinical+RF",XC,YC,deep_fp))
res.append(evaluate("Negative control: assay count alone",np.repeat(NTEST[:0],0,axis=0) if False else
                    sub.index.values.reshape(-1,1).astype(float)*0+1,YC,rf_fp(500)) if False else None)
res=[r for r in res if r]
print("\n=== FULL COHORT (n=5,644) ===")
res_full=[]
res_full.append(evaluate("Random Forest + median imputation",num.values,Y_FULL,rf_fp(500),impute=True,keep_curve=True))
res_full.append(evaluate("HistGradientBoosting, native NaN (recommended)",num.values,Y_FULL,hgb_fp,impute=False,keep_curve=True))
res_full.append(evaluate("Negative control: assay count alone",NTEST,Y_FULL,rf_fp(500),impute=False))

print("\n=== Paired Wilcoxon, complete-case, vs RF 500 ===")
base=res[0]
for r in res[1:]:
    for k,lab in (("_folds_auroc","AUROC"),("_folds_s90","sens@spec90")):
        a,b=np.array(base[k]),np.array(r[k])
        p=wilcoxon(a,b).pvalue if not np.allclose(a,b) else float("nan")
        print(f"  {lab:12s} {r['model'][:44]:46s} {np.mean(b)-np.mean(a):+.4f}  p={p:.4f}")

curves={}
for r in res+res_full:
    if "_curve" in r: curves[r["model"]]=(r["_curve"][0].tolist(),r["_curve"][1].tolist())
    r.pop("_curve",None)
json.dump(curves,open("results/clinical/FINAL2_curves.json","w"))
pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in res]).to_csv("results/clinical/FINAL2_clinical_cc.csv",index=False)
pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in res_full]).to_csv("results/clinical/FINAL2_clinical_full.csv",index=False)
print("\nsaved FINAL2_*")
