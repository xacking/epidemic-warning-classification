import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import os, json, warnings, numpy as np, pandas as pd
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":.7,
                     "xtick.major.width":.7,"ytick.major.width":.7,"figure.dpi":400})
BLUE,ORANGE,PURPLE="#2b6fb5","#c26a1f","#7a4f9e"
INK,MUTED,GRID="#1a1a1a","#5c5c5c","#d8d8d8"
SEQ=LinearSegmentedColormap.from_list("seq",["#f4f8fc","#2b6fb5"])


def despine(ax, keep=("left","bottom")):
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_visible(s in keep); ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3)

# ---------- confusion-matrix panels ----------
def cm_panel(ax, counts, title, total_note):
    tn,fp,fn,tp = counts
    M=np.array([[tn,fp],[fn,tp]],float)
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=M.max())
    for i in range(2):
        for j in range(2):
            v=M[i,j]
            ax.text(j,i,f"{int(round(v)):,}",ha="center",va="center",fontsize=9.5,
                    color=("white" if v>M.max()*.55 else INK),
                    fontweight="bold" if i==j else "normal")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Predicted\nnon-warning","Predicted\nwarning"],fontsize=7.2)
    ax.set_yticks([0,1]); ax.set_yticklabels(["True\nnon-warning","True\nwarning"],fontsize=7.2)
    ax.set_title(title,fontsize=8,pad=6,color=INK)
    for s in ax.spines.values(): s.set_color(MUTED)
    ax.tick_params(length=0, colors=MUTED, labelcolor=INK)

cms=json.load(open("results/social/FINAL_social_cms.json"))
S=pd.read_csv("results/social/FINAL_social_folds.csv")
sel=[("CNN(untrained)+RF, V=100 [E1 as published]","(A) CNN-RF, V=100, untrained\n[configuration as published]"),
     ("CNN+BiLSTM(untrained)+sent+RF, V=100 [E3 as published]","(B) CNN-BiLSTM-RF + sentiment, V=100,\nuntrained [configuration as published]"),
     ("CNN+BiLSTM(trained)+sent+RF, V=5000 [proposed]","(C) CNN-BiLSTM-RF + sentiment,\nV=5000, trained [revised]")]
fig,axes=plt.subplots(1,3,figsize=(7.2,2.6))
for ax,(k,t) in zip(axes,sel):
    a=np.array(cms[k]).sum(0)/2.0
    m=S[S.model==k]
    cm_panel(ax,a,t,f"sensitivity {m.sensitivity.mean():.3f}   precision {m.precision.mean():.3f}   F1 {m.f1.mean():.3f}")
plt.tight_layout(); plt.savefig("results/figures/fig_social_cm.png",bbox_inches="tight",facecolor="white"); plt.close()

# ---------- social model comparison: dot plot, mean +- sd ----------
order=["CNN(untrained)+RF, V=100 [E1 as published]",
       "CNN+BiLSTM(untrained)+sent+RF, V=100 [E3 as published]",
       "CNN+BiLSTM(trained)+sent+RF, V=100",
       "CNN(trained)+RF, V=5000",
       "CNN(trained)+sent+RF, V=5000",
       "CNN+BiLSTM(trained)+sent+RF, V=5000 [proposed]",
       "TF-IDF + logistic regression [reference]"]
short=["CNN-RF, V=100, untrained","CNN-BiLSTM-RF+sent, V=100, untrained",
       "CNN-BiLSTM-RF+sent, V=100, trained",
       "CNN-RF, V=5000, trained","CNN-RF+sent, V=5000, trained",
       "CNN-BiLSTM-RF+sent, V=5000, trained","TF-IDF + logistic regression"]
mets=[("sensitivity","Sensitivity",BLUE),("f1","F1-score",ORANGE),("auroc","AUROC",PURPLE)]
fig,axes=plt.subplots(1,3,figsize=(7.2,2.9),sharey=True)
yy=np.arange(len(order))[::-1]
for ax,(key,lab,col) in zip(axes,mets):
    m=[S[S.model==o][key].mean() for o in order]; sd=[S[S.model==o][key].std() for o in order]
    ax.errorbar(m,yy,xerr=sd,fmt="o",ms=5,color=col,ecolor=col,elinewidth=1.4,capsize=2.5,
                markeredgecolor="white",markeredgewidth=.8,zorder=3)
    for v,y_ in zip(m,yy): ax.text(v,y_+.30,f"{v:.3f}",ha="center",fontsize=6.6,color=INK)
    ax.set_xlabel(lab,fontsize=8,color=INK); ax.grid(axis="x",color=GRID,lw=.5,zorder=0)
    ax.set_axisbelow(True); despine(ax,("left",))
    ax.set_xlim(min(m)-max(sd)-.06, 1.015)
axes[0].set_yticks(yy); axes[0].set_yticklabels(short,fontsize=7.2)
for ax in axes:
    ax.axhline(3.5,color=MUTED,lw=.7,ls=(0,(4,3)))
    ax.axhline(0.5,color=MUTED,lw=.7,ls=(0,(1,2)))
fig.text(0.005,0.545,"as published",fontsize=6.6,color=MUTED,style="italic",rotation=90,va="center")
fig.text(0.005,0.255,"revised",fontsize=6.6,color=MUTED,style="italic",rotation=90,va="center")
plt.tight_layout(); plt.savefig("results/figures/fig_social_compare.png",bbox_inches="tight",facecolor="white"); plt.close()

# ---------- vocabulary sweep ----------
sw=json.load(open("results/social/social_vocab_sweep.json"))
V=[d["vocab"] for d in sw]
fig,ax=plt.subplots(figsize=(3.5,2.6))
tfidf=S[S.model=="TF-IDF + logistic regression [reference]"]
for key,lab,col,mk in (("sensitivity","Sensitivity",BLUE,"o"),("f1","F1-score",ORANGE,"s"),("auroc","AUROC",PURPLE,"^")):
    ax.plot(V,[d[key] for d in sw],marker=mk,ms=5,lw=1.8,color=col,label=lab,
            markeredgecolor="white",markeredgewidth=.8,zorder=3)
    ax.text(V[-1]*1.06,sw[-1][key],f"{sw[-1][key]:.3f}",fontsize=6.8,color=col,va="center")
ax.axhline(tfidf.f1.mean(),color=MUTED,lw=1,ls=(0,(4,3)),zorder=1)
ax.text(105,tfidf.f1.mean()-.019,"TF-IDF + logistic regression reference, F1 = %.3f"%tfidf.f1.mean(),
        fontsize=6.4,color=MUTED)
ax.set_xscale("log"); ax.set_xticks(V); ax.set_xticklabels([str(v) for v in V])
ax.set_xlabel("Tokenizer vocabulary size (log scale)",fontsize=8,color=INK)
ax.set_ylabel("Score",fontsize=8,color=INK)
ax.set_xlim(85,9000); ax.grid(color=GRID,lw=.5,zorder=0); ax.set_axisbelow(True); despine(ax)
ax.legend(frameon=False,fontsize=7,loc="lower right")
plt.tight_layout(); plt.savefig("results/figures/fig_vocab_sweep.png",bbox_inches="tight",facecolor="white"); plt.close()
print("social figures written",flush=True)

# ---------- clinical ----------
C=pd.read_csv("results/clinical/FINAL_clinical_folds.csv")
cc=C[(C.cohort=="complete-case")&(C.rule=="@0.5")]
ccms=json.load(open("results/clinical/FINAL_clinical_cms.json"))
corder=["RF on clinical features only [baseline]","CNN(untrained)+clinical+RF",
        "CNN+BiLSTM(untrained)+clinical+RF [as published]","CNN(trained)+clinical+RF",
        "CNN+BiLSTM(trained)+clinical+RF [proposed]"]
cshort=["RF on clinical features (baseline)","CNN-RF, untrained","CNN-BiLSTM-RF, untrained (as published)",
        "CNN-RF, trained","CNN-BiLSTM-RF, trained (proposed)"]
fig,axes=plt.subplots(1,3,figsize=(7.2,2.6))
csel=[("RF on clinical features only [baseline]","(A) Random Forest on clinical\nfeatures alone [baseline]"),
      ("CNN+BiLSTM(untrained)+clinical+RF [as published]","(B) CNN-BiLSTM-RF, untrained\n[configuration as published]"),
      ("CNN+BiLSTM(trained)+clinical+RF [proposed]","(C) CNN-BiLSTM-RF, trained\n[revised]")]
for ax,(k,t) in zip(axes,csel):
    a=np.array(ccms[f"complete-case|{k}"]).sum(0)/2.0
    m=cc[cc.model==k]
    cm_panel(ax,a,t,f"sensitivity {m.sensitivity.mean():.3f}   AUROC {m.auroc.mean():.3f}")
    ax.set_xticklabels(["Predicted\nnegative","Predicted\npositive"],fontsize=7.2)
    ax.set_yticklabels(["True\nnegative","True\npositive"],fontsize=7.2)
plt.tight_layout(); plt.savefig("results/figures/fig_clinical_cm.png",bbox_inches="tight",facecolor="white"); plt.close()

fig,axes=plt.subplots(1,3,figsize=(7.2,2.4),sharey=True)
yy=np.arange(len(corder))[::-1]
for ax,(key,lab,col) in zip(axes,[("auroc","AUROC",BLUE),("sensitivity","Sensitivity",ORANGE),("f1","F1-score",PURPLE)]):
    m=[cc[cc.model==o][key].mean() for o in corder]; sd=[cc[cc.model==o][key].std() for o in corder]
    ax.errorbar(m,yy,xerr=sd,fmt="o",ms=5,color=col,ecolor=col,elinewidth=1.4,capsize=2.5,
                markeredgecolor="white",markeredgewidth=.8,zorder=3)
    for v,y_ in zip(m,yy): ax.text(v,y_+.28,f"{v:.3f}",ha="center",fontsize=6.6,color=INK)
    ax.set_xlabel(lab,fontsize=8,color=INK); ax.grid(axis="x",color=GRID,lw=.5,zorder=0)
    ax.set_axisbelow(True); despine(ax,("left",))
axes[0].set_yticks(yy); axes[0].set_yticklabels(cshort,fontsize=7.2)
plt.tight_layout(); plt.savefig("results/figures/fig_clinical_compare.png",bbox_inches="tight",facecolor="white"); plt.close()

print("clinical figures written",flush=True)
