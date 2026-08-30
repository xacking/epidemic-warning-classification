import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import os, json, time, warnings, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
tf.get_logger().setLevel("ERROR"); SEED=0; MAXLEN=40; EMB=50

def metrics(yt,yp,prob):
    tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else 0.; spec=tn/(tn+fp) if tn+fp else 0.
    prec=tp/(tp+fp) if tp+fp else 0.; f1=2*prec*sens/(prec+sens) if prec+sens else 0.
    npr=tn/(tn+fn) if tn+fn else 0.; nf1=2*npr*spec/(npr+spec) if npr+spec else 0.
    return dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),accuracy=(tp+tn)/len(yt),
        sensitivity=sens,specificity=spec,precision=prec,f1=f1,
        macro_recall=(sens+spec)/2,macro_f1=(f1+nf1)/2,
        auroc=roc_auc_score(yt,prob),ap=average_precision_score(yt,prob))

# ============================ SOCIAL ============================
df=pd.read_csv("social_prepped.csv"); df["tweet"]=df.tweet.fillna("")
texts=df.tweet.values; y=df.label.values.astype(int); sent=df.sentiment_class.values.astype(int)

def social_feats(Xtr,Xte,V,use_lstm,trained,ytr):
    inp=Input(shape=(MAXLEN,)); emb=Embedding(V+2,EMB)(inp)
    cnn=Concatenate()([GlobalMaxPooling1D()(Conv1D(64,k,activation="relu")(emb)) for k in (3,4,5)])
    feat=Concatenate()([cnn,Bidirectional(LSTM(32))(emb)]) if use_lstm else cnn
    if trained:
        clf=Model(inp,Dense(1,activation="sigmoid")(Dense(32,activation="relu")(Dropout(0.3)(feat))))
        clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
        xa,xv,ya,yv=train_test_split(Xtr,ytr,test_size=.15,random_state=SEED,stratify=ytr)
        cw={0:1.,1:float((ytr==0).sum()/(ytr==1).sum())}
        clf.fit(xa,ya,validation_data=(xv,yv),epochs=15,batch_size=64,verbose=0,class_weight=cw,
                callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
    ext=Model(inp,feat)
    return ext.predict(Xtr,verbose=0,batch_size=256), ext.predict(Xte,verbose=0,batch_size=256)

SOC=[("CNN(untrained)+RF, V=100 [E1 as published]",       dict(V=100, lstm=False,tr=False,s=False)),
     ("CNN+BiLSTM(untrained)+sent+RF, V=100 [E3 as published]",dict(V=100,lstm=True,tr=False,s=True)),
     ("CNN+BiLSTM(trained)+sent+RF, V=100",               dict(V=100, lstm=True, tr=True, s=True)),
     ("CNN(trained)+RF, V=5000",                          dict(V=5000,lstm=False,tr=True, s=False)),
     ("CNN(trained)+sent+RF, V=5000",                     dict(V=5000,lstm=False,tr=True, s=True)),
     ("CNN+BiLSTM(trained)+sent+RF, V=5000 [proposed]",   dict(V=5000,lstm=True, tr=True, s=True)),
     ("TF-IDF + logistic regression [reference]",         dict(kind="tfidf"))]

cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=SEED)
rows=[]; cms={n:[] for n,_ in SOC}; t0=time.time()
for fold,(tr,te) in enumerate(cv.split(texts,y)):
    toks={}
    for V in (100,5000):
        t=Tokenizer(num_words=V,oov_token="<OOV>"); t.fit_on_texts(texts[tr])
        toks[V]=(pad_sequences(t.texts_to_sequences(texts[tr]),padding="post",maxlen=MAXLEN),
                 pad_sequences(t.texts_to_sequences(texts[te]),padding="post",maxlen=MAXLEN))
    for name,c in SOC:
        tf.keras.utils.set_random_seed(SEED+fold)
        if c.get("kind")=="tfidf":
            v=TfidfVectorizer(max_features=5000,ngram_range=(1,2)).fit(texts[tr])
            m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(v.transform(texts[tr]),y[tr])
            prob=m.predict_proba(v.transform(texts[te]))[:,1]
        else:
            Xtr,Xte=toks[c["V"]]
            Ftr,Fte=social_feats(Xtr,Xte,c["V"],c["lstm"],c["tr"],y[tr])
            if c["s"]:
                Ftr=np.hstack([Ftr,sent[tr].reshape(-1,1)]); Fte=np.hstack([Fte,sent[te].reshape(-1,1)])
            sc=StandardScaler().fit(Ftr)
            rf=RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=2).fit(sc.transform(Ftr),y[tr])
            prob=rf.predict_proba(sc.transform(Fte))[:,1]
        r=metrics(y[te],(prob>=.5).astype(int),prob); r.update(model=name,fold=fold)
        rows.append(r); cms[name].append((r["TN"],r["FP"],r["FN"],r["TP"]))
    print("social fold %d  %.0fs"%(fold,time.time()-t0),flush=True)
S=pd.DataFrame(rows); S.to_csv("results/social/FINAL_social_folds.csv",index=False)
S.groupby("model")[["accuracy","sensitivity","specificity","precision","f1","macro_recall","macro_f1","auroc","ap"]].agg(["mean","std"]).to_csv("results/social/FINAL_social_summary.csv")
json.dump({k:[list(map(int,c)) for c in v] for k,v in cms.items()},open("results/social/FINAL_social_cms.json","w"),indent=1)
print(S.groupby("model")[["sensitivity","precision","f1","auroc"]].mean().round(3).to_string(),flush=True)

# ============================ CLINICAL (complete-case cohort) ============================
raw=pd.read_excel("data/covid 19 clinical dataset.xlsx").drop(columns=["Patient ID"]).rename(columns={"SARS-Cov-2 exam result":"labels"})
raw["labels"]=raw["labels"].map({"positive":1,"negative":0})
num=raw.select_dtypes(include=[np.number]).drop(columns=["labels"],errors="ignore")
keep=[c for c in num.columns if num[c].notna().mean()>=0.10]
sub=raw[keep+["labels"]].dropna()
Xc=sub[keep].values.astype(float); yc=sub.labels.values.astype(int)
Xf=pd.read_csv("clinical_X.csv").values.astype(float); yf=np.load("clinical_y.npy").astype(int)
print("\nclinical cohorts: complete-case n=%d f=%d pos=%d | full n=%d f=%d pos=%d"%(
    len(Xc),Xc.shape[1],yc.sum(),len(Xf),Xf.shape[1],yf.sum()),flush=True)
json.dump({"complete_case_features":keep,"n":int(len(Xc)),"pos":int(yc.sum())},open("results/clinical/FINAL_clinical_cohort.json","w"),indent=1)

def clin_feats(A,B,use_lstm,trained,ytr):
    inp=Input(shape=(A.shape[1],1))
    c=GlobalMaxPooling1D()(Conv1D(64,3,activation="relu")(Conv1D(32,3,activation="relu")(inp)))
    feat=Concatenate()([c,Bidirectional(LSTM(32))(inp)]) if use_lstm else c
    a,b=A[...,None],B[...,None]
    if trained:
        clf=Model(inp,Dense(1,activation="sigmoid")(Dense(32,activation="relu")(Dropout(0.3)(feat))))
        clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
        xa,xv,ya,yv=train_test_split(a,ytr,test_size=.15,random_state=SEED,stratify=ytr)
        cw={0:1.,1:float((ytr==0).sum()/(ytr==1).sum())}
        clf.fit(xa,ya,validation_data=(xv,yv),epochs=40,batch_size=32,verbose=0,class_weight=cw,
                callbacks=[EarlyStopping(patience=6,restore_best_weights=True)])
    ext=Model(inp,feat); return ext.predict(a,verbose=0,batch_size=256), ext.predict(b,verbose=0,batch_size=256)

CLIN=[("RF on clinical features only [baseline]",dict(deep=None,raw=True)),
      ("CNN(untrained)+clinical+RF",             dict(deep=(False,False),raw=True)),
      ("CNN+BiLSTM(untrained)+clinical+RF [as published]",dict(deep=(True,False),raw=True)),
      ("CNN(trained)+clinical+RF",               dict(deep=(False,True),raw=True)),
      ("CNN+BiLSTM(trained)+clinical+RF [proposed]",dict(deep=(True,True),raw=True))]

allrows=[]; ccms={}
for cohort,(Xd,yd) in {"complete-case":(Xc,yc),"full-imputed":(Xf,yf)}.items():
    cv2=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=SEED)
    for fold,(tr,te) in enumerate(cv2.split(Xd,yd)):
        imp=SimpleImputer(strategy="median").fit(Xd[tr]); mm=MinMaxScaler().fit(imp.transform(Xd[tr]))
        A,B=mm.transform(imp.transform(Xd[tr])),mm.transform(imp.transform(Xd[te]))
        for name,c in CLIN:
            if cohort=="full-imputed" and c["deep"] is not None and c["deep"][1]: continue
            tf.keras.utils.set_random_seed(SEED+fold)
            ptr_l,pte_l=[],[]
            if c["deep"] is not None:
                a,b=clin_feats(A,B,c["deep"][0],c["deep"][1],yd[tr]); ptr_l.append(a); pte_l.append(b)
            ptr_l.append(A); pte_l.append(B)
            Ftr,Fte=np.hstack(ptr_l),np.hstack(pte_l)
            sc=StandardScaler().fit(Ftr)
            rf=RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=2,class_weight="balanced_subsample").fit(sc.transform(Ftr),yd[tr])
            ptr=rf.predict_proba(sc.transform(Ftr))[:,1]
            pr,rc,th=precision_recall_curve(yd[tr],ptr)
            f1s=np.divide(2*pr*rc,pr+rc,out=np.zeros_like(pr),where=(pr+rc)>0)
            thr=float(th[max(np.argmax(f1s[:-1]),0)]) if len(th) else .5
            prob=rf.predict_proba(sc.transform(Fte))[:,1]
            for tag,t in (("@0.5",.5),("@tuned",thr)):
                r=metrics(yd[te],(prob>=t).astype(int),prob)
                r.update(model=name,fold=fold,cohort=cohort,rule=tag); allrows.append(r)
                if tag=="@tuned": ccms.setdefault((cohort,name),[]).append((r["TN"],r["FP"],r["FN"],r["TP"]))
        print("clinical %s fold %d  %.0fs"%(cohort,fold,time.time()-t0),flush=True)
C=pd.DataFrame(allrows); C.to_csv("results/clinical/FINAL_clinical_folds.csv",index=False)
json.dump({f"{k[0]}|{k[1]}":[list(map(int,x)) for x in v] for k,v in ccms.items()},open("results/clinical/FINAL_clinical_cms.json","w"),indent=1)
for co in ("complete-case","full-imputed"):
    for tag in ("@0.5","@tuned"):
        d=C[(C.cohort==co)&(C.rule==tag)]
        if len(d): print(f"\n--- clinical {co} {tag} ---\n",
            d.groupby("model")[["accuracy","sensitivity","specificity","precision","f1","auroc","ap"]].mean().round(3).to_string(),flush=True)
C.groupby(["cohort","rule","model"])[["accuracy","sensitivity","specificity","precision","f1","macro_recall","macro_f1","auroc","ap"]].agg(["mean","std"]).to_csv("results/clinical/FINAL_clinical_summary.csv")
print("\nALL DONE %.0fs"%(time.time()-t0))
