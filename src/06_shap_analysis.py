import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import os, json, warnings, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"figure.dpi":400})
import tensorflow as tf, shap
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
tf.get_logger().setLevel("ERROR"); SEED=0
tf.keras.utils.set_random_seed(SEED)

meta=json.load(open("results/clinical/FINAL_clinical_cohort.json")); keep=meta["complete_case_features"]
raw=pd.read_excel("data/covid 19 clinical dataset.xlsx").drop(columns=["Patient ID"]).rename(columns={"SARS-Cov-2 exam result":"labels"})
raw["labels"]=raw["labels"].map({"positive":1,"negative":0})
sub=raw[keep+["labels"]].dropna()
X=sub[keep].values.astype(float); y=sub.labels.values.astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
imp=SimpleImputer(strategy="median").fit(Xtr); mm=MinMaxScaler().fit(imp.transform(Xtr))
A,B=mm.transform(imp.transform(Xtr)),mm.transform(imp.transform(Xte))

inp=Input(shape=(A.shape[1],1))
c=GlobalMaxPooling1D()(Conv1D(64,3,activation="relu")(Conv1D(32,3,activation="relu")(inp)))
feat=Concatenate()([c,Bidirectional(LSTM(32))(inp)])
clf=Model(inp,Dense(1,activation="sigmoid")(Dense(32,activation="relu")(Dropout(0.3)(feat))))
clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy")
xa,xv,ya,yv=train_test_split(A[...,None],ytr,test_size=.15,random_state=SEED,stratify=ytr)
cw={0:1.,1:float((ytr==0).sum()/(ytr==1).sum())}
clf.fit(xa,ya,validation_data=(xv,yv),epochs=40,batch_size=32,verbose=0,class_weight=cw,
        callbacks=[EarlyStopping(patience=6,restore_best_weights=True)])
ext=Model(inp,feat)
Ftr=np.hstack([ext.predict(A[...,None],verbose=0),A]); Fte=np.hstack([ext.predict(B[...,None],verbose=0),B])
sc=StandardScaler().fit(Ftr)
rf=RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=2,class_weight="balanced_subsample").fit(sc.transform(Ftr),ytr)
prob=rf.predict_proba(sc.transform(Fte))[:,1]
print("held-out AUROC %.3f"%roc_auc_score(yte,prob),flush=True)

nraw=A.shape[1]
expl=shap.TreeExplainer(rf,feature_perturbation="interventional")
sv=expl.shap_values(sc.transform(Fte),check_additivity=False)
sv_pos = sv[:,:,1] if sv.ndim==3 else sv
raw_block = sv_pos[:,-nraw:]
learned_block = sv_pos[:,:-nraw]
share_raw = np.abs(raw_block).mean(0).sum()/(np.abs(sv_pos).mean(0).sum())
print("share of mean|SHAP| in the raw-clinical block: %.3f"%share_raw,flush=True)

names=[k.replace("\xa0"," ").strip() for k in keep]
shap.summary_plot(raw_block, B, feature_names=names, show=False, max_display=18, plot_size=(6.4,4.6))
f=plt.gcf()
for a in f.axes: a.tick_params(labelsize=7.2); a.set_xlabel(a.get_xlabel(),fontsize=8)
f.savefig("results/figures/fig_shap_beeswarm.png",bbox_inches="tight",facecolor="white",dpi=400); plt.close(f)

# mean|SHAP| bar, drawn directly so the layout is controlled
_o=np.argsort(np.abs(raw_block).mean(0))
_v=np.abs(raw_block).mean(0)[_o]; _n=[names[i] for i in _o]
fig,ax=plt.subplots(figsize=(4.6,3.6))
ax.barh(range(len(_v)),_v,height=.62,color="#2b6fb5")
ax.set_yticks(range(len(_v))); ax.set_yticklabels(_n,fontsize=7)
for i,v in enumerate(_v):
    if v>0.0009: ax.text(v+_v.max()*.015,i,f"{v:.4f}",va="center",fontsize=6.4,color="#1a1a1a")
ax.set_xlabel("mean |SHAP value|",fontsize=8,color="#1a1a1a")
ax.grid(axis="x",color="#d8d8d8",lw=.5); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
for sp in ("left","bottom"): ax.spines[sp].set_color("#5c5c5c")
ax.tick_params(colors="#5c5c5c",labelcolor="#1a1a1a",length=3)
ax.set_xlim(0,_v.max()*1.22)
plt.tight_layout(); plt.savefig("results/figures/fig_shap_bar.png",bbox_inches="tight",facecolor="white"); plt.close()

rank=pd.DataFrame({"feature":names,"mean_abs_shap":np.abs(raw_block).mean(0)}).sort_values("mean_abs_shap",ascending=False)
print("\nSHAP ranking (raw clinical block):"); print(rank.to_string(index=False))
pi=permutation_importance(rf,sc.transform(Fte),yte,scoring="roc_auc",n_repeats=20,random_state=0,n_jobs=2)
pr=pd.DataFrame({"feature":["learned_%d"%i for i in range(Ftr.shape[1]-nraw)]+names,
                 "delta_auc":pi.importances_mean,"sd":pi.importances_std}).sort_values("delta_auc",ascending=False)
print("\nPermutation importance, top 15:"); print(pr.head(15).to_string(index=False))
rank.to_csv("results/clinical/FINAL_shap_ranking.csv",index=False); pr.to_csv("results/clinical/FINAL_permutation_importance.csv",index=False)
json.dump({"held_out_auroc":float(roc_auc_score(yte,prob)),"share_raw_block":float(share_raw),
           "shap_top5":rank.head(5).feature.tolist()},open("results/clinical/FINAL_shap_meta.json","w"),indent=1)
