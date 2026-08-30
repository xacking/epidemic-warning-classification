import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social",): os.makedirs(_d, exist_ok=True)

# Vocabulary-cap sweep for the proposed configuration (CNN+BiLSTM+sentiment+RF,
# extractors trained).  Isolates the effect of the tokenizer's num_words cap,
# which is the single change that accounts for most of the social-media gain.
#
# Five folds (n_repeats=1), not the ten used elsewhere: this sweep measures a
# large effect and the extra repeat costs a further ~8 min for no change in the
# conclusion.  Values therefore differ from the ten-fold table in
# FINAL_social_summary.csv by run-to-run variation of a few thousandths.

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
tf.get_logger().setLevel("ERROR"); SEED=0; MAXLEN=40; EMB=50

def metrics(yt,yp,prob):
    tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else 0.; spec=tn/(tn+fp) if tn+fp else 0.
    prec=tp/(tp+fp) if tp+fp else 0.; f1=2*prec*sens/(prec+sens) if prec+sens else 0.
    return dict(accuracy=(tp+tn)/len(yt),sensitivity=sens,specificity=spec,precision=prec,f1=f1,
                auroc=roc_auc_score(yt,prob),ap=average_precision_score(yt,prob))

df=pd.read_csv("social_prepped.csv"); df["tweet"]=df.tweet.fillna("")
texts=df.tweet.values; y=df.label.values.astype(int); sent=df.sentiment_class.values.astype(int)

cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=1,random_state=SEED)
out=[]; t0=time.time()
for V in (100, 1000, 5000):
    per_fold=[]
    for fold,(tr,te) in enumerate(cv.split(texts,y)):
        tf.keras.utils.set_random_seed(SEED+fold)
        tok=Tokenizer(num_words=V,oov_token="<OOV>"); tok.fit_on_texts(texts[tr])
        Xtr=pad_sequences(tok.texts_to_sequences(texts[tr]),padding="post",maxlen=MAXLEN)
        Xte=pad_sequences(tok.texts_to_sequences(texts[te]),padding="post",maxlen=MAXLEN)
        inp=Input(shape=(MAXLEN,)); emb=Embedding(V+2,EMB)(inp)
        cnn=Concatenate()([GlobalMaxPooling1D()(Conv1D(64,k,activation="relu")(emb)) for k in (3,4,5)])
        feat=Concatenate()([cnn,Bidirectional(LSTM(32))(emb)])
        clf=Model(inp,Dense(1,activation="sigmoid")(Dense(32,activation="relu")(Dropout(0.3)(feat))))
        clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
        xa,xv,ya,yv=train_test_split(Xtr,y[tr],test_size=.15,random_state=SEED,stratify=y[tr])
        cw={0:1.0,1:float((y[tr]==0).sum()/(y[tr]==1).sum())}
        clf.fit(xa,ya,validation_data=(xv,yv),epochs=15,batch_size=64,verbose=0,class_weight=cw,
                callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
        ext=Model(inp,feat)
        Ftr=np.hstack([ext.predict(Xtr,verbose=0,batch_size=256),sent[tr].reshape(-1,1)])
        Fte=np.hstack([ext.predict(Xte,verbose=0,batch_size=256),sent[te].reshape(-1,1)])
        sc=StandardScaler().fit(Ftr)
        rf=RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=2).fit(sc.transform(Ftr),y[tr])
        p=rf.predict_proba(sc.transform(Fte))[:,1]
        per_fold.append(metrics(y[te],(p>=.5).astype(int),p))
    m=pd.DataFrame(per_fold).mean()
    out.append(dict(vocab=V,**{k:round(float(m[k]),3) for k in ("sensitivity","precision","f1","auroc","ap")}))
    print(f"  vocab={V:5d}  sens={m.sensitivity:.3f}  prec={m.precision:.3f}  F1={m.f1:.3f}  AUROC={m.auroc:.3f}  [{time.time()-t0:.0f}s]",flush=True)

json.dump(out,open("results/social/social_vocab_sweep.json","w"),indent=1)
print("wrote results/social/social_vocab_sweep.json",flush=True)
