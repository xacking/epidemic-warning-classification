import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
for _d in ("results/social","results/clinical","results/figures"): os.makedirs(_d, exist_ok=True)

import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.linewidth":.7,"figure.dpi":400})
BLUE,ORANGE,PURPLE="#2b6fb5","#c26a1f","#7a4f9e"; INK,MUTED,GRID="#1a1a1a","#5c5c5c","#d8d8d8"
def despine(ax,keep=("left","bottom")):
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_visible(s in keep); ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED,labelcolor=INK,length=3)

cur=json.load(open("results/clinical/FINAL2_curves.json"))
cc=pd.read_csv("results/clinical/FINAL2_clinical_cc.csv"); fu=pd.read_csv("results/clinical/FINAL2_clinical_full.csv")

# ---------- 1. operating-point figure ----------
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1))
ax=axes[0]
for key,lab,col in (("Random Forest, 500 trees (recommended)","Complete-case cohort (n = 598)",BLUE),
                    ("HistGradientBoosting, native NaN (recommended)","Full cohort (n = 5,644)",ORANGE)):
    y,p=np.array(cur[key][0]),np.array(cur[key][1])
    fpr,tpr,_=roc_curve(y,p)
    ax.plot(fpr,tpr,lw=2,color=col,label=f"{lab}   AUROC {roc_auc_score(y,p):.3f}",zorder=3)
ax.plot([0,1],[0,1],lw=.9,ls=(0,(3,3)),color=MUTED,zorder=1)
r=cc.iloc[0]
for spec,sens,lbl,off in ((0.90,r.s90,"90% specificity","lo"),(0.80,r.s80,"80% specificity","hi")):
    ax.plot(1-spec,sens,marker="o",ms=7,color=BLUE,markeredgecolor="white",markeredgewidth=1.3,zorder=5)
    xy=(14,-30) if off=="lo" else (16,6)
    ax.annotate(f"{lbl}\nsensitivity {sens:.3f}",(1-spec,sens),textcoords="offset points",
                xytext=xy,fontsize=6.8,color=INK,
                arrowprops=dict(arrowstyle="-",lw=.6,color=MUTED,
                                shrinkA=2,shrinkB=4))
ax.set_xlabel("1 − specificity (false-positive rate)",fontsize=8,color=INK)
ax.set_ylabel("Sensitivity",fontsize=8,color=INK)
ax.set_xlim(-.02,1.02); ax.set_ylim(-.02,1.02)
ax.grid(color=GRID,lw=.5,zorder=0); ax.set_axisbelow(True); despine(ax)
ax.legend(frameon=False,fontsize=6.9,loc="lower right")

ax=axes[1]
labs=["Threshold 0.5\n(as reported)","80% specificity","90% specificity"]
vals=[r.sens05,r.s80,r.s90]; errs=[r.sens05_sd,r.s80_sd,r.s90_sd]
cols=[MUTED,BLUE,BLUE]
ax.barh([0,1,2],vals,xerr=errs,height=.5,color=cols,
        error_kw=dict(elinewidth=1.1,capsize=3,ecolor=MUTED),zorder=3)
for v,e,i in zip(vals,errs,[0,1,2]):
    ax.text(v+e+.025,i,f"{v:.3f}",va="center",fontsize=7.6,color=INK)
ax.set_yticks([0,1,2]); ax.set_yticklabels(labs,fontsize=7.4)
ax.set_xlabel("Positive-class sensitivity",fontsize=8,color=INK)
ax.set_xlim(0,1.12); ax.grid(axis="x",color=GRID,lw=.5,zorder=0); ax.set_axisbelow(True)
despine(ax,("left",))
plt.tight_layout(); plt.savefig("results/figures/fig_operating_point.png",bbox_inches="tight",facecolor="white"); plt.close()

# ---------- 2. cohort spectrum ----------
sp=pd.read_csv("results/clinical/IMPROVE_cohort_spectrum.csv").sort_values("threshold")
fig,ax=plt.subplots(figsize=(4.6,2.9))
ax.plot(sp.n,sp.auroc,marker="o",ms=6,lw=1.8,color=BLUE,markeredgecolor="white",
        markeredgewidth=.9,zorder=3)
POS={242:(0,13,"center"),598:(-4,-26,"right"),5644:(-6,16,"right")}
for _,row in sp[sp.n.isin(list(POS))].iterrows():
    dx,dy,ha=POS[int(row.n)]
    ax.annotate(f"{int(row.n_features)} features\nn = {int(row.n):,}",(row.n,row.auroc),
                textcoords="offset points",xytext=(dx,dy),ha=ha,fontsize=6.6,color=MUTED)
ax.axhline(.5,color=MUTED,lw=.9,ls=(0,(3,3)),zorder=1)
ax.text(175,.513,"chance",fontsize=6.4,color=MUTED)
ax.set_xscale("log"); ax.set_xlabel("Records available at that completeness threshold (log scale)",fontsize=7.8,color=INK)
ax.set_ylabel("AUROC",fontsize=8,color=INK)
ax.set_ylim(.45,1.0); ax.set_xlim(170,11000)
ax.grid(color=GRID,lw=.5,zorder=0); ax.set_axisbelow(True); despine(ax)
plt.tight_layout(); plt.savefig("results/figures/fig_cohort_spectrum.png",bbox_inches="tight",facecolor="white"); plt.close()
print("written")
