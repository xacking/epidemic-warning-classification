import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs("results/figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8})
INK,RULE="#1a1a1a","#6b6b6b"
IN,CLEAN,SEQ,AUX,LAB,OUT="#eef2f5","#e3ebf1","#dfe8ef","#f0ece2","#e6e2ee","#e2ece4"

def canvas(w,h,xl,yl):
    fig,ax=plt.subplots(figsize=(w,h)); ax.set_xlim(0,xl); ax.set_ylim(0,yl); ax.axis("off"); return fig,ax
def box(ax,x,y,w,h,t,fill,fs=7.4,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=1.2",
        lw=.9,edgecolor=RULE,facecolor=fill))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=INK,linespacing=1.35,
            fontweight="bold" if bold else "normal")
def arr(ax,x1,y1,x2,y2,dashed=False,lab=None):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=8,lw=.85,
        color=RULE,shrinkA=0,shrinkB=0,linestyle=(0,(3,2)) if dashed else "solid"))
    if lab: ax.text((x1+x2)/2,(y1+y2)/2,lab,fontsize=6.4,color=RULE,ha="center",
                    va="center",bbox=dict(fc="white",ec="none",pad=1.2))

# ---------------- Figure 3: text pipeline ----------------
fig,ax=canvas(6.6,4.9,100,74)
box(ax,26,66,48,7,"Ebola tweet corpus\n13,629 records → 8,395 after exact-duplicate removal",IN)
box(ax,13,52,74,11,"Data cleaning (row-independent, applied once)\n"
  "lowercase · profanity mask · HTML unescape · remove @mentions, #hashtags, URLs\n"
  "keep letters only · drop the retweet marker · remove 179 English stopwords · drop tokens of ≤ 2 characters",CLEAN,fs=6.5)
arr(ax,50,66,50,63)
# three branches
box(ax,1.5,36,28,9,"TextBlob polarity\n→ sentiment class\n(positive / neutral / negative)",AUX)
box(ax,36,36,28,9,"Keras Tokenizer\nfitted on the training fold only\nV = 5,000, reserved OOV index",SEQ)
box(ax,70.5,36,28,9,"Symptom dictionary\n22 terms, substring match",LAB)
for x in (15.5,50,84.5): arr(ax,50,52,x,45)
box(ax,36,23,28,8,"Index sequence\nzero-padded to $L_{max}$ = 40",SEQ)
arr(ax,50,36,50,31)
box(ax,36,11,28,8,"Trainable embedding\n50 dimensions",SEQ)
arr(ax,50,23,50,19)
box(ax,70.5,23,28,8,"Class label $y_i$\n(prediction target)",LAB)
arr(ax,84.5,36,84.5,31)
box(ax,1.5,23,28,8,"Auxiliary scalar feature",AUX)
arr(ax,15.5,36,15.5,31)
box(ax,20,1,60,7,"CNN and Bidirectional LSTM branches → Random Forest",OUT,bold=True)
arr(ax,50,11,50,8)
arr(ax,15.5,23,30,8)
arr(ax,84.5,23,70,8,dashed=True,lab="target, not input")
ax.text(50,-3.2,"Lemmatisation is defined in Equation (5) but is disabled in the released implementation; the label never enters the input vector.",
        ha="center",fontsize=6.4,color="#4a4a4a",style="italic")
plt.tight_layout(pad=.2); plt.savefig("results/figures/fig_workflow_text.png",dpi=400,bbox_inches="tight",facecolor="white"); plt.close()

# ---------------- Figure 4: clinical pipeline ----------------
fig,ax=canvas(6.6,4.2,100,64)
box(ax,22,56,56,7,"COVID-19 clinical records\n5,644 patients × 111 columns",IN)
box(ax,22,44,56,8,"Drop uniformly zero or missing columns (7 removed)\n→ 102 predictors, of which 35 categorical",CLEAN)
arr(ax,50,56,50,52)
box(ax,22,33,56,8,"Encode categorical assays (constant fill + label encoding)\nmap the target to positive = 1 / negative = 0",CLEAN)
arr(ax,50,44,50,41)
box(ax,1.5,17,46,10,"Full cohort\n5,644 records × 102 predictors\n558 positive (9.9%)",SEQ)
box(ax,52.5,17,46,10,"Complete-case cohort\nvariables recorded for ≥ 10% of patients,\nthen records complete on all of them\n598 records × 18 predictors, 81 positive (13.5%)",SEQ,fs=6.9)
arr(ax,50,33,24.5,27); arr(ax,50,33,75.5,27)
box(ax,14,6,72,8,"Inside each training fold only: median imputation → MinMax scaling → classifier\nfive-fold stratified cross-validation, repeated twice",OUT,bold=True)
arr(ax,24.5,17,40,14); arr(ax,75.5,17,60,14)
ax.text(50,1.6,"No feature-selection or correlation-pruning step is applied at any point. Decision thresholds are set from out-of-bag predictions on the training fold.",
        ha="center",fontsize=6.4,color="#4a4a4a",style="italic")
plt.tight_layout(pad=.2); plt.savefig("results/figures/fig_workflow_clinical.png",dpi=400,bbox_inches="tight",facecolor="white"); plt.close()
print("fig3, fig4 rewritten")
