"""UpSet of the 40 paper genes across cell types, from significant age DEGs (lfsr < 0.05).
A gene is in a cell type's set if it is an age DEG in >= 1 louvain x region unit of that cell type.
One figure: up (age_pm > 0) on top, down (age_pm < 0) below. Intersections are exclusive (standard UpSet).
Membership dots only, no connecting lines."""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--outdir", required=True)
a = p.parse_args()

GENE_IDS = {"EZH1": "ENSMMUG00000006093", "EZH2": "ENSMMUG00000013494", "SUZ12": "ENSMMUG00000001116",
    "KDM1A": "ENSMMUG00000009773", "KDM6A": "ENSMMUG00000003317", "KDM6B": "ENSMMUG00000008173",
    "EHMT1": "ENSMMUG00000019613", "EHMT2": "ENSMMUG00000011679", "DOT1L": "ENSMMUG00000010107",
    "KMT2D": "ENSMMUG00000015718", "MEN1": "ENSMMUG00000031160", "DNMT1": "ENSMMUG00000014604",
    "HDAC1": "ENSMMUG00000009948", "USP7": "ENSMMUG00000047388", "CTCF": "ENSMMUG00000009798",
    "LMNB1": "ENSMMUG00000000823", "REST": "ENSMMUG00000018821", "RCOR1": "ENSMMUG00000005845",
    "CTNNB1": "ENSMMUG00000014627", "GSK3B": "ENSMMUG00000001980", "MAPT": "ENSMMUG00000004122",
    "CDKN1A": "ENSMMUG00000055964", "CDKN2B": "ENSMMUG00000001134", "KL": "ENSMMUG00000017048",
    "BDNF": "ENSMMUG00000008634", "PAX6": "ENSMMUG00000016810", "ASCL1": "ENSMMUG00000005747",
    "NEUROG2": "ENSMMUG00000014078", "CDX2": "ENSMMUG00000052070", "HNF4A": "ENSMMUG00000006464",
    "MYOD1": "ENSMMUG00000006400", "FOXA1": "ENSMMUG00000056013", "MTOR": "ENSMMUG00000019325",
    "ACLY": "ENSMMUG00000018931", "ATF4": "ENSMMUG00000055780", "AHR": "ENSMMUG00000012203",
    "TGFB1": "ENSMMUG00000008142", "CGAS": "ENSMMUG00000019606", "STING1": "ENSMMUG00000012048",
    "POMC": "ENSMMUG00000055135"}
if len(GENE_IDS) != 40 or len(set(GENE_IDS.values())) != 40:
    sys.exit("expected 40 unique genes")
ID2GENE = {v: k for k, v in GENE_IDS.items()}
CELL_TYPES = [("glutamatergic neurons", "EXC"), ("GABAergic neurons", "INH"), ("medium spiny neurons", "MSN"),
              ("basket cells", "BC"), ("cerebellar neurons", "CER"), ("midbrain neurons", "MBN"),
              ("astrocytes", "AST"), ("oligodendrocytes", "OLIG"), ("OPC", "OPC"), ("microglia", "MGL"),
              ("vascular cells", "VASC"), ("ependymal cells", "EPEN")]
ABBR = dict(CELL_TYPES)
ROWS = [ab for _, ab in CELL_TYPES]
FS = dict(gene=11, tick=13, label=14, title=16, count=12)

out = os.path.join(a.outdir, "paper40_upset_up_down.png")
if os.path.exists(out):
    sys.exit(f"EXISTS, not overwriting: {out}")
os.makedirs(a.outdir, exist_ok=True)

df = pd.read_csv(a.csv)
missing = set(ABBR) - set(df["cell_type"])
if missing:
    sys.exit(f"cell types not in CSV: {sorted(missing)}")
s = df[df["ensembl_gene_id"].isin(ID2GENE)].copy()
s["gene"] = s["ensembl_gene_id"].map(ID2GENE)
s["ct"] = s["cell_type"].map(ABBR)

res = {}
for d, sign in (("up", 1), ("down", -1)):
    sub = s[np.sign(s["age_pm"]) == sign]
    memb = sub.groupby("gene")["ct"].apply(frozenset)
    inter = {}
    for g, cts in memb.items():
        inter.setdefault(cts, []).append(g)
    inter = sorted(inter.items(), key=lambda kv: (-len(kv[1]), -len(kv[0]), min(ROWS.index(c) for c in kv[0])))
    set_sizes = [sub.loc[sub["ct"] == r, "gene"].nunique() for r in ROWS]
    res[d] = (memb, inter, set_sizes)
    print(f"{d}: {len(memb)} genes in {len(inter)} intersections")
    for cts, gs in inter:
        print(f"  {'+'.join(c for c in ROWS if c in cts):35s} {len(gs):2d}  {', '.join(sorted(gs))}")

nmax = max(len(v[1]) for v in res.values())
ny = len(ROWS)
ymax = max(len(gs) for v in res.values() for _, gs in v[1])
smax = max(max(v[2]) for v in res.values())

fig = plt.figure(figsize=(0.8 * nmax + 5, 24))
gsp = fig.add_gridspec(5, 2, width_ratios=[1.3, max(3, 0.8 * nmax)],
                       height_ratios=[1.6, 1.25, 0.3, 1.6, 1.25], wspace=0.04, hspace=0.06)
for k, (d, color) in enumerate((("up", "tab:red"), ("down", "tab:blue"))):
    memb, inter, set_sizes = res[d]
    r0 = 3 * k
    axb = fig.add_subplot(gsp[r0, 1])
    axm = fig.add_subplot(gsp[r0 + 1, 1], sharex=axb)
    axs = fig.add_subplot(gsp[r0 + 1, 0], sharey=axm)

    x = np.arange(len(inter))
    axb.bar(x, [len(gs) for _, gs in inter], color=color, width=0.6)
    for xi, (_, gs) in zip(x, inter):
        axb.text(xi, len(gs) + 0.15, "\n".join(sorted(gs)), ha="center", va="bottom", fontsize=FS["gene"])
    axb.set_ylim(0, ymax * 2.0 + 2)
    axb.yaxis.set_major_locator(MaxNLocator(integer=True))
    axb.set_ylabel("genes in intersection", fontsize=FS["label"])
    axb.tick_params(labelbottom=False, labelsize=FS["tick"])
    axb.set_title(f"{d.upper()} with age: {len(memb)} of 40 genes", fontsize=FS["title"], loc="left",
                  color=color, fontweight="bold")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    for yi in range(ny):
        if yi % 2 == 0:
            axm.axhspan(yi - 0.5, yi + 0.5, color="0.95", zorder=0)
    for xi, (cts, _) in zip(x, inter):
        ys = [ROWS.index(c) for c in cts]
        axm.scatter([xi] * ny, range(ny), s=80, color="0.85", zorder=1)
        axm.scatter([xi] * len(ys), ys, s=80, color="k", zorder=2)
    axm.set_xlim(-0.6, nmax - 0.4)
    axm.set_ylim(ny - 0.5, -0.5)
    axm.set_xticks([]); axm.tick_params(labelleft=False, left=False)
    for sp in axm.spines.values():
        sp.set_visible(False)

    axs.barh(range(ny), set_sizes, color="0.4", height=0.6)
    for yi, v in enumerate(set_sizes):
        axs.text(v + 0.3, yi, str(v), va="center", ha="right", fontsize=FS["count"])
    axs.set_xlim(smax * 1.3 + 1, 0)
    axs.set_yticks(range(ny)); axs.set_yticklabels(ROWS, fontsize=FS["tick"])
    axs.yaxis.tick_right()
    axs.tick_params(axis="y", length=0, pad=6)
    axs.tick_params(axis="x", labelsize=FS["tick"] - 1)
    axs.set_xlabel("genes per cell type", fontsize=FS["label"] - 1)
    for sp in ("top", "left"):
        axs.spines[sp].set_visible(False)

fig.suptitle("Paper genes among significant age DEGs (lfsr < 0.05 in >= 1 louvain x region unit)\n"
             "bars = genes in exactly that combination of cell types", fontsize=FS["title"] + 1, y=0.995)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
