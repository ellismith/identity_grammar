import argparse, os, sys
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--region", required=True)
p.add_argument("--min_age", type=float, required=True)
p.add_argument("--outdir", required=True)
a = p.parse_args()
if os.path.exists(a.outdir):
    sys.exit(f"EXISTS, not overwriting: {a.outdir}")

H5AD = "/data/CEM/smacklab/data/bican/u01_cell_class/intermediate_cell_class/Res1_GABAergic-neurons_subset.h5ad"
GENELIST = "/scratch/easmit31/prc2/paper_gene_overlap.tsv"
REGION = a.region

gl = pd.read_csv(GENELIST, sep="\t")
gl = gl[gl["n_ensembl"] == 1][["gene", "source", "ensembl_ids"]].rename(columns={"ensembl_ids": "ensembl_id"})
gl = pd.concat([gl, pd.DataFrame([{"gene": "AHR", "source": "ref_title_only",
                                   "ensembl_id": "ENSMMUG00000012203"}])], ignore_index=True)
if len(gl) != 40 or gl["gene"].duplicated().any():
    sys.exit(f"Expected 40 unique genes, got {len(gl)}")
GENES = gl["gene"].tolist()

A = ad.read_h5ad(H5AD, backed="r")
gidx = A.var_names.get_indexer(gl["ensembl_id"])
if (gidx < 0).any():
    sys.exit(f"IDs not in h5ad: {list(gl['ensembl_id'][gidx < 0])}")
regions = A.obs["region"].astype(str).values
if REGION not in set(regions):
    sys.exit(f"{REGION} not found. Region values: {sorted(set(regions))}")
os.makedirs(a.outdir)

keep = np.flatnonzero(regions == REGION)
print(f"{REGION}: {len(keep):,} of {A.n_obs:,} cells; {len(GENES)} genes", flush=True)
parts = []
for s in range(0, A.n_obs, 100000):
    e = min(s + 100000, A.n_obs)
    rows = keep[(keep >= s) & (keep < e)] - s
    if len(rows) == 0:
        continue
    X = A.X[s:e][rows][:, gidx]
    parts.append(X.toarray() if sparse.issparse(X) else np.asarray(X))
X = np.vstack(parts)

obs = A.obs.iloc[keep]
cells = pd.DataFrame(X, columns=GENES)
cells["animal_id"] = obs["animal_id"].astype(str).values
cells["age"] = pd.to_numeric(obs["age"].astype(str)).values
cells["sex"] = obs["sex"].astype(str).values
cells = cells[cells["age"] >= a.min_age]

pb = cells.groupby("animal_id").agg(age=("age", "first"), sex=("sex", "first"), n_cells=("age", "size"))
pb = pb.join(cells.groupby("animal_id")[GENES].mean())
pb = pb.join(cells[GENES].gt(0).groupby(cells["animal_id"]).mean().add_suffix("_frac"))
pb = pb.sort_values("age")
pb.to_csv(os.path.join(a.outdir, "per_animal.tsv"), sep="\t")
print(f"min_age={a.min_age}: {len(pb)} animals, ages {pb['age'].min():.2f}-{pb['age'].max():.2f}, "
      f"{(pb['age'] < 1).sum()} under age 1")
print(f"cells per animal: min {pb['n_cells'].min()}, median {pb['n_cells'].median():.0f}")

sex_term = " + C(sex)" if pb["sex"].nunique() > 1 else ""
stats = []
for g, src, eid in gl[["gene", "source", "ensembl_id"]].itertuples(index=False):
    y = pb[g]
    row = dict(region=REGION, min_age=a.min_age, gene=g, source=src, ensembl_id=eid,
               n_animals=len(pb), mean_frac=pb[f"{g}_frac"].mean(), mean_expr=y.mean(),
               pearson_r=np.nan, pearson_p=np.nan, spearman_rho=np.nan, spearman_p=np.nan,
               ols_beta_age=np.nan, ols_se_age=np.nan, ols_p_age=np.nan)
    if y.std() > 0:
        row["pearson_r"], row["pearson_p"] = pearsonr(pb["age"], y)
        row["spearman_rho"], row["spearman_p"] = spearmanr(pb["age"], y)
        m = smf.ols("y ~ age" + sex_term, data=pb.assign(y=y)).fit()
        row.update(ols_beta_age=m.params["age"], ols_se_age=m.bse["age"], ols_p_age=m.pvalues["age"])
    stats.append(row)
stats = pd.DataFrame(stats)
stats.to_csv(os.path.join(a.outdir, "stats.tsv"), sep="\t", index=False)
print(f"genes with no expression in this region: {list(stats.gene[stats.mean_expr == 0])}")

nc, nr = 5, 8
fig, axes = plt.subplots(nr, nc, figsize=(16, 22))
inf = pb["age"] < 1
for ax, row in zip(axes.flat, stats.itertuples()):
    g = row.gene
    ax.scatter(pb.loc[~inf, "age"], pb.loc[~inf, g], s=10, c="k")
    ax.scatter(pb.loc[inf, "age"], pb.loc[inf, g], s=10, c="tab:red")
    if pb[g].std() > 0:
        b = np.polyfit(pb["age"], pb[g], 1)
        xs = np.linspace(pb["age"].min(), pb["age"].max(), 2)
        ax.plot(xs, np.polyval(b, xs), c="tab:blue")
    ax.set_title(f"{g}  rho={row.spearman_rho:.2f}", fontsize=9)
    ax.tick_params(labelsize=7)
fig.suptitle(f"GABAergic neurons, {REGION}, min_age={a.min_age}  (red = age < 1; y = per-animal mean log-norm expr)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(a.outdir, "paper40_vs_age.png"), dpi=110)
print(f"Done: {a.outdir}")
