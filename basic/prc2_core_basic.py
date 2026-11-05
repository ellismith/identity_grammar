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
os.makedirs(a.outdir)

H5AD = "/data/CEM/smacklab/data/bican/u01_cell_class/intermediate_cell_class/Res1_GABAergic-neurons_subset.h5ad"
REGION = a.region
GENES = ["EZH1", "EZH2", "EED", "SUZ12"]

A = ad.read_h5ad(H5AD, backed="r")
regions = A.obs["region"].astype(str).values
if REGION not in set(regions):
    sys.exit(f"{REGION} not found. Region values: {sorted(set(regions))}")
sym = A.var["external_gene_name"].astype(str).values
gidx = []
for g in GENES:
    hits = np.flatnonzero(sym == g)
    if len(hits) != 1:
        sys.exit(f"{g}: {len(hits)} matches in external_gene_name")
    gidx.append(hits[0])

keep = np.flatnonzero(regions == REGION)
print(f"{REGION}: {len(keep):,} of {A.n_obs:,} cells", flush=True)
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

stats = []
fml_sex = pb["sex"].nunique() > 1
for g in GENES:
    r, rp = pearsonr(pb["age"], pb[g])
    rho, sp = spearmanr(pb["age"], pb[g])
    m = smf.ols(f"{g} ~ age" + (" + C(sex)" if fml_sex else ""), pb).fit()
    stats.append(dict(region=REGION, min_age=a.min_age, gene=g, n_animals=len(pb),
                      mean_frac=pb[f"{g}_frac"].mean(),
                      pearson_r=r, pearson_p=rp, spearman_rho=rho, spearman_p=sp,
                      ols_beta_age=m.params["age"], ols_se_age=m.bse["age"], ols_p_age=m.pvalues["age"]))
stats = pd.DataFrame(stats)
stats.to_csv(os.path.join(a.outdir, "stats.tsv"), sep="\t", index=False)
print(stats.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, g, row in zip(axes, GENES, stats.itertuples()):
    inf = pb["age"] < 1
    ax.scatter(pb.loc[~inf, "age"], pb.loc[~inf, g], s=15, c="k")
    ax.scatter(pb.loc[inf, "age"], pb.loc[inf, g], s=15, c="tab:red")
    b = np.polyfit(pb["age"], pb[g], 1)
    xs = np.linspace(pb["age"].min(), pb["age"].max(), 2)
    ax.plot(xs, np.polyval(b, xs), c="tab:blue")
    ax.set_title(f"{g}\nr={row.pearson_r:.2f} rho={row.spearman_rho:.2f} ols p={row.ols_p_age:.2g}", fontsize=9)
    ax.set_xlabel("age")
axes[0].set_ylabel("mean log-norm expr (per animal)")
fig.suptitle(f"GABAergic neurons, {REGION}, min_age={a.min_age}  (red = age < 1)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(a.outdir, "prc2_core_vs_age.png"), dpi=130)
print(f"Done: {a.outdir}")
