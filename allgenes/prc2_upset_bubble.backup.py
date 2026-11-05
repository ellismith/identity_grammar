"""UpSet plots of a gene set among significant age DEGs (age_lfsr < 0.05), up (age_pm > 0) on top,
down (age_pm < 0) below. Columns = exclusive intersections, top bars = genes per intersection,
left bars = genes per row. Each dot is SIZED by the number of significant louvain subtypes.
Each panel shows only rows with >= 1 hit in that direction.
--level celltype   : rows = cell types; dot size = n distinct subtypes (any region)
--level ctregion   : rows = cell type x region; dot size = n subtypes in that region
--level subtype    : one figure per cell type, rows = louvain subtypes; dot size = n REGIONS
--level bycategory : for EACH gene category, two figures: cell type level (as celltype) and
                     subtype level (rows = all louvain subtypes of all cell types; dot size = n regions)
--level percelltype: for EACH cell type, two figures with all genes:
                     region level (rows = regions; dot size = n subtypes in that region) and
                     subtype x region level (rows = louvain | region units; one dot per significant unit)
Gene names are stacked one per line and colored by category (category column of the gene file).
Multi-gene columns: dot size = MEAN per gene (per-gene counts in the _intersections.tsv)."""
import argparse, os, sys, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--level", required=True, choices=["celltype", "ctregion", "subtype", "bycategory", "percelltype"])
p.add_argument("--genes", required=True, help="pipe-separated: category|gene|ensembl_id|alias_role|source")
p.add_argument("--lfsr", type=float, default=0.05)
a = p.parse_args()
os.makedirs(a.outdir, exist_ok=True)

GENES_DF = pd.read_csv(a.genes, sep="|")
GENES = list(GENES_DF[["category", "gene", "ensembl_id"]].itertuples(index=False, name=None))
ID2G = {i: g for _, g, i in GENES}
PAL = ["#7b3fa0","#0090c0","#d95f02","#00897b","#a8760a","#3b6fb6","#c2185b","#5e8f1e","#9a6fb0","#b03a2e"]  # validated text palette
CATS = list(dict.fromkeys(GENES_DF.category))
if len(CATS) > len(PAL): print(f"WARNING: {len(CATS)} categories > {len(PAL)} colors; colors repeat")
CATCOL = {c: PAL[i % len(PAL)] for i, c in enumerate(CATS)}
GCOL = {g: CATCOL[c] for c, g, _ in GENES}
NG = len(GENES)
if len(ID2G) != NG: sys.exit("duplicate IDs in gene list")

AB = {"glutamatergic neurons":"EXC","GABAergic neurons":"INH","medium spiny neurons":"MSN","basket cells":"BC",
      "cerebellar neurons":"CER","midbrain neurons":"MBN","astrocytes":"AST","oligodendrocytes":"OLIG","OPC":"OPC",
      "microglia":"MGL","vascular cells":"VASC","ependymal cells":"EPEN"}
CTS = list(AB.values())
REG = ["ACC","dlPFC","M1","IPP","EC","HIP","mdTN","CN","NAc","MB","lCb"]

df = pd.read_csv(a.csv)
need = {"ensembl_gene_id","cell_type","louvain","region","age_lfsr","age_pm"}
if not need <= set(df.columns): sys.exit(f"CSV missing columns: {need - set(df.columns)}")
df = df[(df.age_lfsr < a.lfsr) & df.ensembl_gene_id.isin(ID2G)].copy()
unk = set(df.cell_type) - set(AB)
if unk: sys.exit(f"unknown cell types: {unk}")
df["gene"] = df.ensembl_gene_id.map(ID2G); df["ct"] = df.cell_type.map(AB)
df["dir"] = np.where(df.age_pm > 0, "up", "down")
extra = sorted(set(df.region) - set(REG)); REG += extra
print(f"{len(df)} significant rows for {df.gene.nunique()} of {NG} genes")

def make(sub, rowkey, rows, sizefun, title, out, rowgroup, ng=None, unit="subtypes"):
    ng = ng or NG
    panels = {}
    for d in ["up", "down"]:
        s = sub[sub.dir == d]
        if s.empty:
            panels[d] = (pd.DataFrame(columns=["gene", rowkey, "n"]), {}, []); continue
        cnt = s.groupby(["gene", rowkey]).apply(sizefun).rename("n").reset_index()
        memb = cnt.groupby("gene")[rowkey].apply(lambda x: tuple(sorted(x, key=rows.index)))
        cols = {}
        for g, key in memb.items(): cols.setdefault(key, []).append(g)
        order = sorted(cols, key=lambda k: (-len(cols[k]), -len(k), rows.index(k[0])))
        panels[d] = (cnt, cols, order)
    def colsize(gs, r, look):
        return float(np.mean([int(look.get((g, r), 0)) for g in gs]))
    sizes = [colsize(cols[k], r, c.set_index(["gene", rowkey]).n) for c, cols, o in panels.values() for k in o for r in k]
    maxn = max(sizes + [1])
    RH = 0.26 if len(rows) > 15 else 0.38
    def area(n): return 8 + ((RH * 72 * 0.92) ** 2 - 8) * (n / maxn)
    ncol = max([len(o) for _, _, o in panels.values()] + [1])
    prow = {d: ([r for r in rows if r in set(c[rowkey])] or rows[:1]) for d, (c, _, _) in panels.items()}
    CW = 0.62; LEFT = 2.6; RIGHT = 0.9
    W = max(9.0, LEFT + CW * ncol + RIGHT)
    maxg = max([len(cols[k]) for _, cols, o in panels.values() for k in o] + [1])
    LABH = 0.14 * maxg + 0.35; BARH = 1.2; GAP = 0.9; TITLEH = 2.7
    PH = {d: LABH + BARH + RH * len(prow[d]) + 0.5 for d in prow}
    H = TITLEH + PH["up"] + PH["down"] + GAP
    fig = plt.figure(figsize=(W, H))
    def ax(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])
    rows_tsv = []
    for pi, (d, color) in enumerate([("up", "#d62728"), ("down", "#1f77b4")]):
        cnt, cols, order = panels[d]
        rws = prow[d]; nrow = len(rws); MATH = RH * nrow
        ybase = H - TITLEH - PH["up"] + 0.5 if d == "up" else 0.5
        mx, mw = LEFT, CW * ncol
        axm = ax(mx, ybase, mw, MATH); axb = ax(mx, ybase + MATH + 0.05, mw, BARH)
        axt = ax(mx, ybase + MATH + 0.05 + BARH, mw, LABH); axt.axis("off")
        axl = ax(0.25, ybase, 1.2, MATH)
        axt.set_title(f"{d.upper()} with age: {cnt.gene.nunique()} of {ng} genes", loc="left", color=color, fontweight="bold", fontsize=13)
        grp = [rowgroup(r) for r in rws]; ug = list(dict.fromkeys(grp))
        for i in range(nrow):
            if ug.index(grp[i]) % 2 == 0: axm.axhspan(i - 0.5, i + 0.5, color="#efefef", zorder=0, lw=0)
        look = cnt.set_index(["gene", rowkey]).n if len(cnt) else pd.Series(dtype=float)
        for j, key in enumerate(order):
            gs = cols[key]
            axm.scatter([j] * nrow, range(nrow), s=10, color="#d9d9d9", zorder=1, lw=0)
            for r in key:
                axm.scatter(j, rws.index(r), s=area(colsize(gs, r, look)), color="black", zorder=2, lw=0)
            rows_tsv.append({"direction": d, "column": j + 1, "genes": ",".join(gs), "n_genes": len(gs),
                             "rows (per-gene counts)": "; ".join(f"{r}: " + ",".join(f"{g}={int(look.get((g, r), 0))}" for g in gs) for r in key)})
        if not order: axm.text(ncol / 2, nrow / 2, "no genes in this direction", ha="center", color="grey")
        axm.set_xlim(-0.6, ncol - 0.4); axm.set_ylim(nrow - 0.5, -0.5)
        axm.set_yticks(range(nrow)); axm.set_yticklabels(rws, fontsize=9); axm.set_xticks([])
        for sp in axm.spines.values(): sp.set_visible(False)
        axm.tick_params(axis="y", length=0)
        hts = [len(cols[k]) for k in order]
        axb.bar(range(len(order)), hts, color=color, width=0.6)
        axb.set_xlim(-0.6, ncol - 0.4); axb.set_ylim(0, max(hts + [1]) * 1.05); axb.set_xticks([])
        axb.set_ylabel("genes in\nintersection", fontsize=9); axb.spines[["top", "right"]].set_visible(False)
        axb.yaxis.get_major_locator().set_params(integer=True)
        axt.set_xlim(-0.6, ncol - 0.4); axt.set_ylim(0, (LABH - 0.35) / 0.14 + 2.2)
        for j, k in enumerate(order):
            for li, g in enumerate(reversed(cols[k])):
                axt.text(j, 0.2 + li, g, ha="center", va="bottom", fontsize=7.5, color=GCOL[g], fontweight="bold")
        per = cnt.groupby(rowkey).gene.nunique().reindex(rws).fillna(0) if len(cnt) else pd.Series(0, index=rws)
        axl.barh(range(nrow), per.values, color="#666666", height=0.6)
        for i, v in enumerate(per.values):
            if v: axl.text(v + 0.03 * max(per.max(), 1), i, int(v), ha="right", va="center", fontsize=8)
        axl.set_ylim(nrow - 0.5, -0.5); axl.set_xlim(max(per.max(), 1) * 1.25, 0); axl.set_yticks([])
        axl.spines[["top", "left", "right"]].set_visible(False); axl.set_xlabel("genes per row", fontsize=8)
    ref = sorted(set([1, max(1, int(round(maxn / 2))), int(np.floor(maxn))]))
    hs = [Line2D([], [], ls="", marker="o", color="black", markersize=np.sqrt(area(n)), label=f"{n}") for n in ref]
    fig.legend(handles=hs, title=f"dot size = n {unit}", loc="upper left", bbox_to_anchor=(0.25 / W, 1 - 1.6 / H),
               ncol=len(ref), frameon=False, fontsize=9, title_fontsize=9, handletextpad=0.6, columnspacing=2.0)
    fig.text(0.25 / W, 1 - 0.15 / H, title, ha="left", va="top", fontsize=12)
    shown = {g for _, cols, o in panels.values() for k in o for g in cols[k]}
    used = [c for c in CATS if c in set(GENES_DF[GENES_DF.gene.isin(shown)].category)]
    ch = [Line2D([], [], ls="", marker="s", markersize=8, color=CATCOL[c], label=c) for c in used]
    fig.legend(handles=ch, loc="upper left", bbox_to_anchor=(0.25 / W, 1 - 0.62 / H), ncol=min(len(ch), 5),
               frameon=False, fontsize=9, labelcolor="linecolor", handletextpad=0.3, columnspacing=1.2, title="gene category", title_fontsize=9) if ch else None
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows_tsv).to_csv(out.replace(".png", "_intersections.tsv"), sep="\t", index=False)
    print("saved", out)

base = f"{os.path.basename(a.genes)} ({NG} genes) among significant age DEGs (lfsr < {a.lfsr})"
if a.level == "celltype":
    rows = [c for c in CTS if c in set(df.ct)]
    make(df, "ct", rows, lambda g: g.louvain.nunique(),
         base + "\nrows = cell types with >= 1 hit in that direction; dot size = n louvain subtypes significant in that direction (any region)",
         os.path.join(a.outdir, "prc2genes_upset_celltype.png"), lambda r: r)
elif a.level == "ctregion":
    df["ctreg"] = df.ct + " | " + df.region
    rows = [f"{c} | {r}" for c in CTS for r in REG if f"{c} | {r}" in set(df.ctreg)]
    make(df, "ctreg", rows, lambda g: g.louvain.nunique(),
         base + "\nrows = cell type x regions with >= 1 hit in that direction; dot size = n louvain subtypes significant in that region and direction",
         os.path.join(a.outdir, "prc2genes_upset_ctregion.png"), lambda r: r.split(" | ")[0])
elif a.level == "bycategory":
    for cat in CATS:
        gset = set(GENES_DF.gene[GENES_DF.category == cat]); ngc = len(gset)
        s = df[df.gene.isin(gset)].copy()
        slug = re.sub(r"[^A-Za-z0-9]+", "_", cat).strip("_")
        if s.empty: print(f"skip {cat}: no hits"); continue
        cb = f"{cat}: {ngc} genes among significant age DEGs (lfsr < {a.lfsr})"
        rows = [c for c in CTS if c in set(s.ct)]
        make(s, "ct", rows, lambda g: g.louvain.nunique(),
             cb + "\nrows = cell types with >= 1 hit in that direction; dot size = n louvain subtypes significant in that direction (any region)",
             os.path.join(a.outdir, f"cat_{slug}_celltype.png"), lambda r: r, ng=ngc)
        s["sub"] = s.ct + "_" + s.louvain.astype(str)
        rows = [f"{c}_{l}" for c in CTS for l in sorted(s.louvain[s.ct == c].unique())]
        make(s, "sub", rows, lambda g: g.region.nunique(),
             cb + "\nrows = louvain subtypes with >= 1 hit in that direction; dot size = n regions where that subtype is significant in that direction",
             os.path.join(a.outdir, f"cat_{slug}_subtype.png"), lambda r: r.rsplit("_", 1)[0], ng=ngc, unit="regions")
elif a.level == "percelltype":
    for c in CTS:
        s = df[df.ct == c].copy()
        if s.empty: print(f"skip {c}: no hits"); continue
        rows = [r for r in REG if r in set(s.region)]
        make(s, "region", rows, lambda g: g.louvain.nunique(),
             base + f"\n{c}: rows = regions with >= 1 hit in that direction; dot size = n louvain subtypes significant in that region and direction",
             os.path.join(a.outdir, f"percelltype_{c}_region.png"), lambda r: r)
        s["unit"] = s.region + " | " + s.louvain.astype(str)
        rows = [f"{r} | {l}" for r in REG for l in sorted(s.louvain[s.region == r].unique())]
        make(s, "unit", rows, lambda g: 1,
             base + f"\n{c}: rows = subtype (louvain) x region units with >= 1 hit in that direction; one dot per significant unit",
             os.path.join(a.outdir, f"percelltype_{c}_subtype_region.png"), lambda r: r.split(" | ")[0], unit="units")
else:
    for c in CTS:
        s = df[df.ct == c].copy()
        if s.empty: print(f"skip {c}: no hits"); continue
        s["sub"] = c + "_" + s.louvain.astype(str)
        rows = [f"{c}_{l}" for l in sorted(s.louvain.unique())]
        make(s, "sub", rows, lambda g: g.region.nunique(),
             base + f"\n{c}: rows = louvain subtypes with >= 1 hit in that direction; dot size = n regions where that subtype is significant in that direction",
             os.path.join(a.outdir, f"prc2genes_upset_subtype_{c}.png"), lambda r: r, unit="regions")
