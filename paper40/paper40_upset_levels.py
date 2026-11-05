"""UpSet of the 40 paper genes from significant age DEGs (lfsr < 0.05), at finer levels.
  --level celltype_region : one figure, sets = cell type x region combos (rows with >= 1 paper-gene hit)
  --level subtype_region  : one figure for --cell_type, sets = louvain x region units (rows with >= 1 hit)
Each figure: up (age_pm > 0) on top, down (age_pm < 0) below, same rows in both panels.
Intersections are exclusive (standard UpSet). Membership dots only, no connecting lines."""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--level", required=True, choices=["celltype_region", "subtype_region"])
p.add_argument("--cell_type", help="required for --level subtype_region")
a = p.parse_args()
if a.level == "subtype_region" and not a.cell_type:
    sys.exit("--cell_type is required for --level subtype_region")

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
CT_ORDER = [ab for _, ab in CELL_TYPES]
REGION_ORDER = ["ACC", "dlPFC", "M1", "IPP", "EC", "HIP", "CN", "NAc", "mdTN", "MB", "lCb"]
FS = dict(gene=11, tick=12, label=14, title=16, count=11)


def rkey(r):
    return REGION_ORDER.index(r) if r in REGION_ORDER else len(REGION_ORDER)


df = pd.read_csv(a.csv)
s = df[df["ensembl_gene_id"].isin(ID2GENE)].copy()
s["gene"] = s["ensembl_gene_id"].map(ID2GENE)

if a.level == "celltype_region":
    out = os.path.join(a.outdir, "paper40_upset_celltype_region.png")
    s["ct"] = s["cell_type"].map(ABBR)
    if s["ct"].isna().any():
        sys.exit(f"unexpected cell types: {sorted(s.loc[s['ct'].isna(), 'cell_type'].unique())}")
    s["row"] = s["ct"] + " | " + s["region"]
    keys = s[["ct", "region", "row"]].drop_duplicates()
    keys = keys.assign(o1=keys["ct"].map(CT_ORDER.index), o2=keys["region"].map(rkey)).sort_values(["o1", "o2"])
    rows = keys["row"].tolist()
    group_of = dict(zip(keys["row"], keys["ct"]))
    row_label, what = "cell type x region", "all cell types"
else:
    if a.cell_type not in set(df["cell_type"]):
        sys.exit(f"cell type {a.cell_type!r} not in CSV. Values: {sorted(df['cell_type'].unique())}")
    slug = a.cell_type.replace(" ", "_")
    out = os.path.join(a.outdir, f"paper40_upset_subtype_region_{slug}.png")
    s = s[s["cell_type"] == a.cell_type].copy()
    if s.empty:
        sys.exit(f"{a.cell_type}: no paper-gene hits, no figure made")
    s["row"] = s["region"] + " | L" + s["louvain"].astype(str)
    keys = s[["region", "louvain", "row"]].drop_duplicates()
    keys = keys.assign(o1=keys["region"].map(rkey)).sort_values(["o1", "region", "louvain"])
    rows = keys["row"].tolist()
    group_of = dict(zip(keys["row"], keys["region"]))
    row_label, what = "louvain x region unit", a.cell_type

if os.path.exists(out):
    sys.exit(f"EXISTS, not overwriting: {out}")
os.makedirs(a.outdir, exist_ok=True)

res = {}
for d, sign in (("up", 1), ("down", -1)):
    sub = s[np.sign(s["age_pm"]) == sign]
    memb = sub.groupby("gene")["row"].apply(frozenset)
    inter = {}
    for g, rs in memb.items():
        inter.setdefault(rs, []).append(g)
    inter = sorted(inter.items(), key=lambda kv: (-len(kv[1]), -len(kv[0]), min(rows.index(r) for r in kv[0])))
    set_sizes = [sub.loc[sub["row"] == r, "gene"].nunique() for r in rows]
    res[d] = (memb, inter, set_sizes)
    print(f"{d}: {len(memb)} genes in {len(inter)} intersections")
    for rs, gs in inter:
        print(f"  {' + '.join(r for r in rows if r in rs)}  ->  {len(gs)}: {', '.join(sorted(gs))}")

ny = len(rows)
nmax = max(max(len(v[1]) for v in res.values()), 1)
ymax = max([len(gs) for v in res.values() for _, gs in v[1]] or [1])
smax = max(max(v[2]) for v in res.values())
mat_h = max(1.25, 0.12 * ny)
w_bar, w_lab, w_mat = 1.6, 0.13 * max(len(r) for r in rows) + 0.3, max(3, 0.8 * nmax)
fig = plt.figure(figsize=(w_bar + w_lab + w_mat + 1, 2 * (3.2 + mat_h * 2.2) + 2))
gsp = fig.add_gridspec(5, 3, width_ratios=[w_bar, w_lab, w_mat],
                       height_ratios=[1.6, mat_h, 0.35, 1.6, mat_h], wspace=0, hspace=0.06)
bands = []
for r in rows:
    if not bands or bands[-1][0] != group_of[r]:
        bands.append([group_of[r], rows.index(r), rows.index(r)])
    else:
        bands[-1][2] = rows.index(r)

for k, (d, color) in enumerate((("up", "tab:red"), ("down", "tab:blue"))):
    memb, inter, set_sizes = res[d]
    r0 = 3 * k
    axb = fig.add_subplot(gsp[r0, 2])
    axm = fig.add_subplot(gsp[r0 + 1, 2], sharex=axb)
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

    for i, (_, lo, hi) in enumerate(bands):
        if i % 2 == 0:
            axm.axhspan(lo - 0.5, hi + 0.5, color="0.94", zorder=0)
            axs.axhspan(lo - 0.5, hi + 0.5, color="0.94", zorder=0)
    for xi, (rs, _) in zip(x, inter):
        ys = [rows.index(r) for r in rs]
        axm.scatter([xi] * ny, range(ny), s=55, color="0.85", zorder=1)
        axm.scatter([xi] * len(ys), ys, s=55, color="k", zorder=2)
    axm.set_xlim(-0.6, nmax - 0.4)
    axm.set_ylim(ny - 0.5, -0.5)
    axm.set_xticks([]); axm.tick_params(labelleft=False, left=False)
    for sp in axm.spines.values():
        sp.set_visible(False)

    axs.barh(range(ny), set_sizes, color="0.4", height=0.6, zorder=1)
    for yi, v in enumerate(set_sizes):
        if v:
            axs.text(v + 0.2, yi, str(v), va="center", ha="right", fontsize=FS["count"])
    axs.set_xlim(smax * 1.35 + 1, 0)
    axs.set_yticks(range(ny)); axs.set_yticklabels(rows, fontsize=FS["tick"])
    axs.yaxis.tick_right()
    axs.tick_params(axis="y", length=0, pad=6)
    axs.tick_params(axis="x", labelsize=FS["tick"] - 1)
    axs.set_xlabel(f"genes per {row_label}", fontsize=FS["label"] - 1)
    for sp in ("top", "left"):
        axs.spines[sp].set_visible(False)

fig.suptitle(f"Paper genes among significant age DEGs (lfsr < 0.05): {what}, by {row_label}\n"
             f"rows = {ny} {row_label}s with >= 1 paper-gene hit; bars = genes in exactly that combination",
             fontsize=FS["title"] + 1, y=0.995)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
