"""mean_vs_disp_ctregion.py

Compare age effects on MEAN expression (u01_age_degs.csv, mashr age_pm/age_lfsr) with age effects on
DISPERSION (DGLM q0.2 tree, mash_beta/mash_lfsr) for a gene set (default: 66-gene PRC2 list),
at cell type x region resolution.

Each condition (cell type x region) gets a profile = one value per gene:
  mean side : mean age_pm across louvain subtypes that are significant (age_lfsr < --mean_lfsr)
              in that cell type x region; 0 if no significant subtype (the DEG table only holds
              significant rows, so "missing" = not significant).
  disp side : same rule on mash_beta / mash_lfsr (< --disp_lfsr). If the DGLM table is already at
              cell type x region level (one row per gene per region), it is used as-is.
Read-only on all inputs. Exits without writing anything if --outdir already exists.

Outputs (in --outdir):
  profile_mean.tsv, profile_disp.tsv     conditions x genes, values as above
  n_sig_subtypes_mean.tsv / _disp.tsv    how many subtypes went into each value
  similarity_mean.tsv, similarity_disp.tsv   condition x condition Pearson r across genes
  similarity_heatmaps.png                (A) mean clustered (B) dispersion clustered
                                         (C) dispersion in the row order of (A)
  mantel.txt                             agreement between the two similarity matrices
  within_condition_agreement.tsv         per condition: r(mean profile, disp profile) across genes
  scatter_by_gene.png                    one panel per gene: x = mean effect, y = disp effect,
                                         one dot per condition, colored by cell type
  scatter_points.tsv                     the numbers behind the scatter
  run_log.txt                            everything printed during the run
"""
import argparse, os, re, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr

P = argparse.ArgumentParser()
P.add_argument("--degs", default="/scratch/easmit31/u01/u01_age_degs.csv")
P.add_argument("--genes", default="/scratch/easmit31/prc2/allgenes/prc2_genes_v2.txt")
P.add_argument("--dglm_tree", default="/scratch/easmit31/dispersion/dglm/disp_age__mask1_915_300c_q0.2")
P.add_argument("--dglm_sub", default="dglm_checkpoints_cutoff0.5/master_dglm_combined.tsv")
P.add_argument("--outdir", default="/scratch/easmit31/prc2/allgenes/mean_vs_disp_ctregion")
P.add_argument("--mean_lfsr", type=float, default=0.05)
P.add_argument("--disp_lfsr", type=float, default=0.05)
P.add_argument("--min_genes", type=int, default=3,
               help="a condition needs >= this many nonzero genes on a side to enter that side's similarity")
P.add_argument("--n_perm", type=int, default=2000)
a = P.parse_args()

if os.path.exists(a.outdir):
    sys.exit(f"ERROR: {a.outdir} already exists. Not overwriting. Pass a new --outdir.")

LOG = []
def log(*x):
    s = " ".join(str(i) for i in x); print(s); LOG.append(s)

ABBR = {"astrocytes": "AST", "microglia": "MGL", "opc": "OPC", "oligodendrocyteprecursorcells": "OPC",
        "oligodendrocytes": "OLIG", "ependymalcells": "EPEN", "vascularcells": "VASC",
        "glutamatergicneurons": "EXC", "gabaergicneurons": "INH", "mediumspinyneurons": "MSN",
        "cerebellarneurons": "CER", "midbrainneurons": "MBN", "basketcells": "BC"}
ORDER = ["EXC", "INH", "MSN", "CER", "MBN", "BC", "AST", "OLIG", "OPC", "MGL", "VASC", "EPEN"]
COL = dict(zip(ORDER, ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
                       "#CC79A7", "#999999", "#882255", "#44AA99", "#117733", "#332288"]))

def abbr(ct):
    k = re.sub(r"[_\- ]\d+$", "", str(ct))          # strip trailing subcluster number
    k = re.sub(r"[^a-z]", "", k.lower())
    if k.upper() in ORDER: return k.upper()
    return ABBR.get(k)

# ---------- gene set ----------
g = pd.read_csv(a.genes, sep="|", dtype=str).fillna("")
need = {"category", "gene", "ensembl_id"}
if need - set(g.columns):
    sys.exit(f"ERROR: gene file missing columns {need - set(g.columns)}; has {list(g.columns)}")
g = g.drop_duplicates("gene")
ens2gene = {r.ensembl_id: r.gene for r in g.itertuples() if r.ensembl_id}
sym2gene = {r.gene: r.gene for r in g.itertuples()}
GENES = list(g.gene)
gcat = dict(zip(g.gene, g.category))
log(f"gene set: {len(GENES)} genes ({len(ens2gene)} with Ensembl IDs) from {a.genes}")

def map_genes(df, ens_col, sym_col):
    gene = df[ens_col].map(ens2gene) if ens_col in df else pd.Series(np.nan, index=df.index)
    if sym_col in df:
        gene = gene.fillna(df[sym_col].map(sym2gene))
    return gene

# ---------- mean side ----------
d = pd.read_csv(a.degs, usecols=lambda c: c in
                {"ensembl_gene_id", "external_gene_name", "cell_type", "louvain", "region", "age_pm", "age_lfsr"})
log(f"DEG table: {len(d):,} rows; max age_lfsr = {d.age_lfsr.max():.3g}")
d["gene"] = map_genes(d, "ensembl_gene_id", "external_gene_name")
d = d[d.gene.notna() & (d.age_lfsr < a.mean_lfsr)].copy()
d["ct"] = d.cell_type.map(abbr)
if d.ct.isna().any():
    log("WARN unmapped DEG cell types (dropped):", sorted(d.loc[d.ct.isna(), "cell_type"].unique()))
d = d[d.ct.notna()]
d["cond"] = d.ct + "|" + d.region.astype(str)
log(f"mean side: {len(d):,} significant gene x subtype x region rows; "
    f"{d.gene.nunique()} of {len(GENES)} genes hit")
mean_val = d.groupby(["cond", "gene"]).age_pm.mean().unstack()
mean_n = d.groupby(["cond", "gene"]).size().unstack()

# ---------- dispersion side ----------
files = sorted(glob.glob(os.path.join(a.dglm_tree, "*", a.dglm_sub)))
if not files:
    sys.exit(f"ERROR: no files matching {os.path.join(a.dglm_tree, '*', a.dglm_sub)}")
log(f"DGLM: {len(files)} cell-type files under {a.dglm_tree}")
parts = []
for f in files:
    parent = os.path.relpath(f, a.dglm_tree).split(os.sep)[0]
    ct = abbr(parent)
    if ct is None:
        log(f"WARN cannot map DGLM folder '{parent}' to a cell type, skipped"); continue
    hdr = pd.read_csv(f, sep="\t", nrows=0).columns
    miss = {"region", "mash_beta", "mash_lfsr"} - set(hdr)
    if miss:
        sys.exit(f"ERROR: {f} missing columns {miss}; has {list(hdr)}")
    keep = [c for c in ["ensembl_id", "symbol", "cell_type", "region", "mash_beta", "mash_lfsr"] if c in hdr]
    chunks = []
    for ch in pd.read_csv(f, sep="\t", usecols=keep, chunksize=500_000):
        ch["gene"] = map_genes(ch, "ensembl_id", "symbol")
        chunks.append(ch[ch.gene.notna()])
    x = pd.concat(chunks)
    x["ct"] = ct
    nsub = x.cell_type.nunique() if "cell_type" in x else 1
    log(f"  {parent:24s} -> {ct:4s}: {len(x):,} gene-set rows, {x.region.nunique()} regions, "
        f"{nsub} cell_type label(s) e.g. {list(x.cell_type.unique()[:3]) if 'cell_type' in x else '-'}")
    parts.append(x)
x = pd.concat(parts)
x["cond"] = x.ct + "|" + x.region.astype(str)
for t in sorted({a.disp_lfsr, 0.05, 0.2}):
    log(f"  dispersion gene-set rows with mash_lfsr < {t}: {(x.mash_lfsr < t).sum():,} of {len(x):,}")
xs = x[x.mash_lfsr < a.disp_lfsr]
disp_val = xs.groupby(["cond", "gene"]).mash_beta.mean().unstack()
disp_n = xs.groupby(["cond", "gene"]).size().unstack()
disp_all = x.groupby(["cond", "gene"]).mash_beta.mean().unstack()   # unfiltered, for the scatter only

# ---------- align ----------
r_mean = set(d.region.astype(str)); r_disp = set(x.region.astype(str))
log(f"regions in DEG table: {sorted(r_mean)}")
log(f"regions in DGLM:      {sorted(r_disp)}")
if r_mean != r_disp:
    log(f"WARN region names differ. only-DEG: {sorted(r_mean - r_disp)}  only-DGLM: {sorted(r_disp - r_mean)}")

all_conds = sorted(set(x.cond) | set(d.cond),
                   key=lambda c: (ORDER.index(c.split("|")[0]), c.split("|")[1]))
def full(m): return m.reindex(index=all_conds, columns=GENES).fillna(0.0)
PM, PD = full(mean_val), full(disp_val)
os.makedirs(a.outdir)
PM.to_csv(f"{a.outdir}/profile_mean.tsv", sep="\t")
PD.to_csv(f"{a.outdir}/profile_disp.tsv", sep="\t")
full(mean_n).astype(int).to_csv(f"{a.outdir}/n_sig_subtypes_mean.tsv", sep="\t")
full(disp_n).astype(int).to_csv(f"{a.outdir}/n_sig_subtypes_disp.tsv", sep="\t")

def sim(P_):
    keep = P_.index[(P_ != 0).sum(axis=1) >= a.min_genes]
    S = P_.loc[keep].T.corr()
    return S
SM, SD = sim(PM), sim(PD)
log(f"conditions entering similarity (>= {a.min_genes} nonzero genes): mean {len(SM)}, dispersion {len(SD)}")
SM.to_csv(f"{a.outdir}/similarity_mean.tsv", sep="\t")
SD.to_csv(f"{a.outdir}/similarity_disp.tsv", sep="\t")

def order(S):
    if len(S) < 3: return list(S.index)
    D = (1 - S.fillna(0).values).clip(0, 2); np.fill_diagonal(D, 0)
    return list(S.index[leaves_list(linkage(squareform(D, checks=False), "average"))])

# ---------- Mantel ----------
shared = sorted(set(SM.index) & set(SD.index))
mtxt = []
if len(shared) >= 4:
    A, B = SM.loc[shared, shared].values, SD.loc[shared, shared].values
    iu = np.triu_indices(len(shared), 1)
    r0 = spearmanr(A[iu], B[iu]).correlation
    rng = np.random.default_rng(1); cnt = 0
    for _ in range(a.n_perm):
        p = rng.permutation(len(shared))
        if spearmanr(A[iu], B[p][:, p][iu]).correlation >= r0: cnt += 1
    pval = (cnt + 1) / (a.n_perm + 1)
    mtxt = [f"Mantel test, Spearman r between condition-similarity matrices (mean vs dispersion)",
            f"shared conditions: {len(shared)}", f"r = {r0:.3f}", f"one-sided permutation p = {pval:.4g} ({a.n_perm} perms)",
            "r > 0: conditions that look alike in mean age effects also look alike in dispersion age effects."]
else:
    mtxt = [f"Mantel test skipped: only {len(shared)} conditions present on both sides."]
open(f"{a.outdir}/mantel.txt", "w").write("\n".join(mtxt) + "\n")
for m in mtxt: log(m)

# ---------- within-condition agreement ----------
rows = []
for c in all_conds:
    u, v = PM.loc[c], PD.loc[c]
    both = ((u != 0) & (v != 0)).sum()
    r = pearsonr(u, v)[0] if (u.std() > 0 and v.std() > 0) else np.nan
    same = int(((np.sign(u) == np.sign(v)) & (u != 0) & (v != 0)).sum())
    rows.append(dict(condition=c, n_genes_mean=int((u != 0).sum()), n_genes_disp=int((v != 0).sum()),
                     n_genes_both=int(both), n_same_direction=same, r_mean_vs_disp=r))
pd.DataFrame(rows).to_csv(f"{a.outdir}/within_condition_agreement.tsv", sep="\t", index=False)

# ---------- heatmaps ----------
def hm(ax, S, idx, title):
    S = S.reindex(index=idx, columns=idx)
    im = ax.imshow(S.values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(idx))); ax.set_yticks(range(len(idx)))
    fs = max(3, min(8, 400 / max(len(idx), 1)))
    ax.set_xticklabels(idx, rotation=90, fontsize=fs); ax.set_yticklabels(idx, fontsize=fs)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(COL.get(lab.get_text().split("|")[0], "k"))
    ax.set_title(title, fontsize=11)
    return im
oM, oD = order(SM), order(SD)
oC = [c for c in oM if c in SD.index]
n = max(len(oM), len(oD), 10)
fig, axs = plt.subplots(1, 3, figsize=(3 * min(0.18 * n + 2, 22), min(0.18 * n + 2, 22)))
hm(axs[0], SM, oM, f"A. Mean age effects\n({len(oM)} conditions, clustered)")
hm(axs[1], SD, oD, f"B. Dispersion age effects\n({len(oD)} conditions, clustered)")
im = hm(axs[2], SD, oC, "C. Dispersion, in A's order\n(shared conditions only)")
fig.colorbar(im, ax=axs, shrink=0.5, label="Pearson r of gene profiles")
fig.suptitle(f"Condition similarity across {len(GENES)} PRC2-list genes  |  {mtxt[2] if len(mtxt) > 2 else mtxt[0]}",
             fontsize=12)
fig.savefig(f"{a.outdir}/similarity_heatmaps.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- per-gene scatter ----------
DA = disp_all.reindex(index=all_conds, columns=GENES)
pts = []
for gname in GENES:
    for c in all_conds:
        yv = DA.loc[c, gname]
        if pd.isna(yv): continue
        pts.append(dict(gene=gname, category=gcat[gname], condition=c, cell_type=c.split("|")[0],
                        mean_effect=PM.loc[c, gname], disp_effect_all=yv,
                        mean_sig=PM.loc[c, gname] != 0, disp_sig=PD.loc[c, gname] != 0))
pts = pd.DataFrame(pts)
pts.to_csv(f"{a.outdir}/scatter_points.tsv", sep="\t", index=False)
show = [gn for gn in GENES if gn in set(pts.gene) and pts[pts.gene == gn][["mean_sig", "disp_sig"]].any(axis=None)]
log(f"scatter: {len(show)} genes with a significant effect on at least one side")
if show:
    nc = 6; nr = int(np.ceil(len(show) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 2.6, nr * 2.4), squeeze=False)
    for ax, gn in zip(axs.flat, show):
        q = pts[pts.gene == gn]
        for _, r in q.iterrows():
            filled = r.mean_sig and r.disp_sig
            ax.scatter(r.mean_effect, r.disp_effect_all, s=14,
                       facecolor=COL[r.cell_type] if filled else "none",
                       edgecolor=COL[r.cell_type], alpha=0.9 if (r.mean_sig or r.disp_sig) else 0.25, lw=0.8)
        ax.axhline(0, c="k", lw=0.4); ax.axvline(0, c="k", lw=0.4)
        ax.set_title(f"{gn} ({gcat[gn]})", fontsize=7); ax.tick_params(labelsize=6)
    for ax in list(axs.flat)[len(show):]: ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COL[k], label=k) for k in ORDER if k in set(pts.cell_type)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=False)
    fig.supxlabel("mean age effect (avg age_pm of sig subtypes; 0 = not sig)", fontsize=9)
    fig.supylabel("dispersion age effect (avg mash_beta, all subclusters)", fontsize=9)
    fig.text(0.5, -0.01, "filled = significant on both sides; open = one side; faint = neither", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{a.outdir}/scatter_by_gene.png", dpi=200, bbox_inches="tight"); plt.close(fig)

open(f"{a.outdir}/run_log.txt", "w").write("\n".join(LOG) + "\n")
log(f"done -> {a.outdir}")
