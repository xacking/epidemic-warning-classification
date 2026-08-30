import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import os, warnings, numpy as np, pandas as pd, json
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input,Embedding,Conv1D,GlobalMaxPooling1D,Bidirectional,LSTM,Dense,Concatenate,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
tf.get_logger().setLevel("ERROR"); SEED=0; tf.keras.utils.set_random_seed(SEED)
from common import SYMPTOMS
df=pd.read_csv("social_prepped.csv"); df["tweet"]=df.tweet.fillna("")
texts=df.tweet.values; y=df.label.values.astype(int); sent=df.sentiment_class.values.astype(int)
idx=np.arange(len(y))
tr,te=train_test_split(idx,test_size=.2,random_state=SEED,stratify=y)
V=5000
tok=Tokenizer(num_words=V,oov_token="<OOV>"); tok.fit_on_texts(texts[tr])
Xtr=pad_sequences(tok.texts_to_sequences(texts[tr]),padding="post",maxlen=40)
Xte=pad_sequences(tok.texts_to_sequences(texts[te]),padding="post",maxlen=40)
inp=Input(shape=(40,)); emb=Embedding(V+2,50)(inp)
cnn=Concatenate()([GlobalMaxPooling1D()(Conv1D(64,k,activation="relu")(emb)) for k in (3,4,5)])
feat=Concatenate()([cnn,Bidirectional(LSTM(32))(emb)])
clf=Model(inp,Dense(1,activation="sigmoid")(Dense(32,activation="relu")(Dropout(.3)(feat))))
clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
xa,xv,ya,yv=train_test_split(Xtr,y[tr],test_size=.15,random_state=SEED,stratify=y[tr])
cw={0:1.,1:float((y[tr]==0).sum()/(y[tr]==1).sum())}
clf.fit(xa,ya,validation_data=(xv,yv),epochs=15,batch_size=64,verbose=0,class_weight=cw,
        callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
ext=Model(inp,feat)
Ftr=np.hstack([ext.predict(Xtr,verbose=0,batch_size=256),sent[tr].reshape(-1,1)])
Fte=np.hstack([ext.predict(Xte,verbose=0,batch_size=256),sent[te].reshape(-1,1)])
sc=StandardScaler().fit(Ftr)
rf=RandomForestClassifier(n_estimators=50,random_state=42,n_jobs=2).fit(sc.transform(Ftr),y[tr])
yp=rf.predict(sc.transform(Fte))
fn=te[(y[te]==1)&(yp==0)]; tp=te[(y[te]==1)&(yp==1)]; fp=te[(y[te]==0)&(yp==1)]
sub=df.iloc[fn]
def kwcount(t): return sum(1 for k in SYMPTOMS if k in t.lower())
neg_words=("not ","no ","dont","doesnt","never","without")
out=dict(n_test=len(te), n_pos=int((y[te]==1).sum()), FN=len(fn), FP=len(fp), TP=len(tp),
    fn_neutral_pct=round(100*float((sub.sentiment_class==2).mean()),1),
    fn_single_kw_pct=round(100*float((sub.tweet.apply(kwcount)==1).mean()),1),
    fn_mean_tokens=round(float(sub.tweet.str.split().apply(len).mean()),1),
    tp_mean_tokens=round(float(df.iloc[tp].tweet.str.split().apply(len).mean()),1),
    fn_negation_pct=round(100*float(sub.tweet.apply(lambda t: any(w in t.lower() for w in neg_words)).mean()),1),
    all_pos_neutral_pct=round(100*float((df[df.label==1].sentiment_class==2).mean()),1))
print(json.dumps(out,indent=1))
json.dump(out,open("results/social/FINAL_error_analysis.json","w"),indent=1)
print("\nsample false negatives:")
for t in sub.tweet.head(8): print("  -",t[:110])
