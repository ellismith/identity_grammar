"""One figure per cell type: paper genes (rows) x louvain-region units (columns),
age_pm of significant age DEGs, up (age_pm > 0) and down (age_pm < 0) as two side-by-side panels."""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--cell_type", required=True)
p.add_argument("--outdir", required=True)
a = p.parse_args()

GROUPS = [
    ("PRC2", [("EZH1", "ENSMMUG00000006093"), ("EZH2", "ENSMMUG00000013494"), ("SUZ12", "ENSMMUG00000001116")]),
    ("Chromatin partners /\nopposers", [("KDM1A", "ENSMMUG00000009773"), ("KDM6A", "ENSMMUG00000003317"),
        ("KDM6B", "ENSMMUG00000008173"), ("EHMT1", "ENSMMUG00000019613"), ("EHMT2", "ENSMMUG00000011679"),
        ("DOT1L", "ENSMMUG00000010107"), ("KMT2D", "ENSMMUG00000015718"), ("MEN1", "ENSMMUG00000031160"),
        ("DNMT1", "ENSMMUG00000014604"), ("HDAC1", "ENSMMUG00000009948"), ("USP7", "ENSMMUG00000047388"),
        ("CTCF", "ENSMMUG00000009798"), ("LMNB1", "ENSMMUG00000000823")]),
    ("REST / β-catenin axis", [("REST", "ENSMMUG00000018821"), ("RCOR1", "ENSMMUG00000005845"),
        ("CTNNB1", "ENSMMUG00000014627"), ("GSK3B", "ENSMMUG00000001980"), ("MAPT", "ENSMMUG00000004122")]),
    ("Senescence /\nlongevity", [("CDKN1A", "ENSMMUG00000055964"), ("CDKN2B", "ENSMMUG00000001134"),
        ("KL", "ENSMMUG00000017048"), ("BDNF", "ENSMMUG00000008634")]),
    ("Lineage TFs", [("PAX6", "ENSMMUG00000016810"), ("ASCL1", "ENSMMUG00000005747"), ("NEUROG2", "ENSMMUG00000014078"),
        ("CDX2", "ENSMMUG00000052070"), ("HNF4A", "ENSMMUG00000006464"), ("MYOD1", "ENSMMUG00000006400"),
        ("FOXA1", "ENSMMUG00000056013")]),
    ("Metabolism / stress /\nsignaling", [("MTOR", "ENSMMUG00000019325"), ("ACLY", "ENSMMUG00000018931"),
        ("ATF4", "ENSMMUG00000055780"), ("AHR", "ENSMMUG00000012203"), ("TGFB1", "ENSMMUG00000008142"),
        ("CGAS", "ENSMMUG00000019606"), ("STING1", "ENSMMUG00000012048"), ("POMC", "ENSMMUG00000055135")]),
]
GENES = [g for _, gs in GROUPS for g, _ in gs]
ID2GENE = {i: g for _, gs in GROUPS for g, i in gs}
if len(GENES) != 40 or len(set(GENES)) != 40:
    sys.exit(f"expected 40 unique genes, got {len(GENES)}")

df = pd.read_csv(a.csv)
if a.cell_type not in set(df["cell_type"]):
    sys.exit(f"cell type {a.cell_type!r} not in CSV. Values: {sorted(df['cell_type'].unique())}")
slug = a.cell_type.replace(" ", "_")
out = os.path.join(a.outdir, f"paper40_age_degs_{slug}.png")
if os.path.exists(out):
    sys.exit(f"EXISTS, not overwriting: {out}")
os.makedirs(a.outdir, exist_ok=True)

ct = df[df["cell_type"] == a.cell_type]
units = (ct[["region", "louvain"]].drop_duplicates().sort_values(["region", "louvain"]).reset_index(drop=True))
col_of = {(r, l): j for j, (r, l) in enumerate(units.itertuples(index=False))}
hits = ct[ct["ensembl_gene_id"].isin(ID2GENE)].copy()
hits["gene"] = hits["ensembl_gene_id"].map(ID2GENE)

M = np.full((len(GENES), len(units)), np.nan)
for h in hits.itertuples():
    M[GENES.index(h.gene), col_of[(h.region, h.louvain)]] = h.age_pm
up = np.where(M > 0, M, np.nan)
dn = np.where(M < 0, -M, np.nan)
vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
print(f"{a.cell_type}: {len(units)} louvain x region units, {hits['gene'].nunique()} paper genes hit, "
      f"{int((M > 0).sum())} up, {int((M < 0).sum())} down")

w = max(6, 0.11 * len(units) + 2.5)
fig, axes = plt.subplots(1, 2, figsize=(2 * w + 2.5, 11), sharey=True)
reg_bounds = units.groupby("region").apply(lambda d: (d.index.min(), d.index.max())).to_dict()
for ax, data, cmap, title in [(axes[0], up, "Reds", "UP with age (age_pm > 0)"),
                              (axes[1], dn, "Blues", "DOWN with age (|age_pm|, age_pm < 0)")]:
    ax.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(units)))
    ax.set_xticklabels(units["louvain"].astype(str), fontsize=5, rotation=90)
    for r, (lo, hi) in reg_bounds.items():
        if lo > 0:
            ax.axvline(lo - 0.5, color="0.3", lw=0.8)
        ax.text((lo + hi) / 2, -1.2, r, ha="center", va="bottom", fontsize=8, fontweight="bold")
    y = 0
    for _, gs in GROUPS:
        if y:
            ax.axhline(y - 0.5, color="k", lw=0.8)
        y += len(gs)
    ax.set_xlabel("louvain subcluster (grouped by region)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=30)
    ax.set_xlim(-0.5, len(units) - 0.5)
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.3, pad=0.08, label="|age_pm|")
axes[0].set_yticks(range(len(GENES)))
axes[0].set_yticklabels(GENES, fontsize=8)
y = 0
for name, gs in GROUPS:
    axes[0].annotate(name, xy=(0, y + len(gs) / 2 - 0.5), xycoords=("axes fraction", "data"), xytext=(-55, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=8, fontweight="bold")
    y += len(gs)
fig.suptitle(f"{a.cell_type}: paper genes among significant age DEGs (lfsr < 0.05)\n"
             f"columns = all {len(units)} louvain x region units with any age DEG; blank = not significant / not tested; "
             f"shared color scale (max |age_pm| = {vmax:.4f})", fontsize=10)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
