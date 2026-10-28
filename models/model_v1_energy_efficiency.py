"""
HE3027 Group Project: are energy-efficient homes less affected by outdoor air pollution?

Hypothesis: energy-efficient homes (better insulation, double glazing, fewer draughts) let in less outdoor air,
so the air inside them depends less on the PM10 outside. If buyers know this, the price discount for PM10 should
be SMALLER for efficient homes, i.e. the interaction PM10 × energy efficiency should be POSITIVE.

Energy efficiency comes from Energy Performance Certificates (EPC). Every home that is sold or let in England
needs one (since 2008). It gives a rating band from A (most efficient) to G (least efficient). Our sales data
have a postcode but no house number, so we cannot find the certificate of the home that was sold. Instead we
describe the homes AROUND it: every certificate is placed at its address point and assigned to the nearest
postcode centroid, and we take the average rating of the homes in the postcode. Flats and houses are kept apart
(homes that share an address point with other homes are flats in the same building), so a flat gets the average
rating of the flats in its postcode and a house the average rating of the houses.

Builds on the v1 model (M8: LSOA and year-quarter fixed effects, MSOA-clustered standard errors) and adds
  1. PM10 × postcode energy efficiency (continuous: per +10 points of the EPC score, about one rating band)
  2. PM10 × efficiency group: efficient (average C or better), average (D, base group), inefficient (E to G)
with checks for the main confounders (flats and new builds are more efficient AND have a different PM10 effect).

Writes model_v1_energy_efficiency.md and figures/energy_efficiency.png. The postcode-level EPC values are cached
in data/epc_postcode.csv (+ data/epc_postcode_info.json), which are built on the first run (delete them to rebuild).

Run from the repository root:
    python models/model_v1_energy_efficiency.py

Needs: pandas, numpy, matplotlib, scipy, shapely, pyproj, pyarrow
Data:  data/land_registry_pollution.dta
       data/epc_addresscloud/<borough>.parquet   (33 files, one per London borough)
           EPC ratings of England and Wales, copy by Addresscloud on Source Cooperative (Open Government Licence):
           https://source.coop/addresscloud/epc  (folder geoparquet-local-authority)
       data/os_codepoint_open/codepo_gb.zip      (keep zipped)
           OS Code-Point Open, CSV, GB: https://osdatahub.os.uk/downloads/open/CodePointOpen
"""

import io
import json
import math
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from model_v1 import (ACCESS, AMENITY, CLUSTER, CODE_DIR, MODEL_DIR, DATA_DIR, PROPERTY, SOCIO, fe_ols, load_data, pct,
                      stars, two_fe_ols)
from model_v1_income_groups import gbp, p_fmt, p_normal, wald
from model_v1_income_small_area import lincom
from model_v1_pm10_smoothing import add_smoothed_pm10
from model_v2 import by_year, coef, sig_txt, varying

MD_FILE = MODEL_DIR / "model_v1_energy_efficiency.md"
FIG_DIR = CODE_DIR / "figures"
FIG_FILE = FIG_DIR / "energy_efficiency.png"

EPC_DIR = DATA_DIR / "epc_addresscloud"
CODEPOINT_ZIP = DATA_DIR / "os_codepoint_open" / "codepo_gb.zip"
EPC_FILE = DATA_DIR / "epc_postcode.csv"
EPC_INFO = DATA_DIR / "epc_postcode_info.json"

CONTROLS = PROPERTY + ACCESS + AMENITY + SOCIO     # the M8 controls

# EPC rating bands and the SAP energy efficiency score behind them (1-100+, higher = more efficient).
# The data only have the band, so each band gets the middle of its score range.
BANDS = {"A": (92, 100), "B": (81, 91), "C": (69, 80), "D": (55, 68), "E": (39, 54), "F": (21, 38), "G": (1, 20)}
BAND_SCORE = {b: (lo + hi) / 2 for b, (lo, hi) in BANDS.items()}

MAX_DIST_M = 200     # a certificate further than this from every postcode centroid is dropped (geocoding errors)
MIN_HOMES = 3        # a sale is used only if at least this many homes of its type have an EPC in its postcode
LONDON_BOX = (495_000, 570_000, 150_000, 210_000)   # British National Grid (easting, northing) around London

GROUPS = {  # key: (label, average score from, to); base group: average (D)
    "eg": ("Inefficient (E-G)", 0, 55),
    "d": ("Average (D)", 55, 69),
    "ac": ("Efficient (A-C)", 69, 101),
}
COLORS = {"eg": "#eb6834", "d": "#8a8984", "ac": "#1baf7a"}


# ---------------------------------------------------------------------------
# Data: EPC ratings per postcode
# ---------------------------------------------------------------------------
def read_epc():
    """One row per home (UPRN): mean EPC score of its certificates, location, and whether it is a flat.

    The copy of the EPC register has every certificate since 2008, but no date. A home with several certificates
    gets the mean of their scores. A home that shares its address point (to the metre) with other homes is a
    flat in a multi-unit building; all other homes are counted as houses.
    """
    files = sorted(EPC_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No EPC files in {EPC_DIR}, see the README for the download.")
    d = pd.concat([pd.read_parquet(f, columns=["UPRN", "CURRENT_ENERGY_RATING", "geometry"]) for f in files],
                  ignore_index=True)
    info = {"boroughs": len(files), "certificates": len(d)}
    d["score"] = d["CURRENT_ENERGY_RATING"].map(BAND_SCORE)
    d = d.dropna(subset=["score"])
    pts = shapely.from_wkb(d["geometry"].to_numpy())
    d["e"], d["n"] = Transformer.from_crs(4326, 27700, always_xy=True).transform(shapely.get_x(pts),
                                                                                   shapely.get_y(pts))
    d = d[np.isfinite(d["e"]) & np.isfinite(d["n"])]
    homes = d.groupby("UPRN").agg(score=("score", "mean"), e=("e", "first"), n=("n", "first"),
                                  certs=("score", "size"))
    homes["flat"] = (homes.groupby([homes["e"].round(0), homes["n"].round(0)])["score"].transform("size") > 1)
    info.update(certificates_valid=len(d), homes=len(homes), homes_multi_cert=int((homes["certs"] > 1).sum()),
                flat_share=float(homes["flat"].mean()))
    return homes, info


def read_codepoint():
    """Postcode centroids (British National Grid) from OS Code-Point Open, only the area around London."""
    cols = ["pc", "pq", "e", "n", "cy", "rh", "lh", "cc", "dc", "wc"]
    with zipfile.ZipFile(CODEPOINT_ZIP) as z:
        cp = pd.concat([pd.read_csv(io.BytesIO(z.read(f)), header=None, names=cols, usecols=["pc", "e", "n"])
                        for f in z.namelist() if f.startswith("Data/CSV/") and f.endswith(".csv")],
                       ignore_index=True)
    e0, e1, n0, n1 = LONDON_BOX
    cp = cp[cp["e"].between(e0, e1) & cp["n"].between(n0, n1)].reset_index(drop=True)
    cp["postcode"] = cp["pc"].str.replace(" ", "", regex=False)
    return cp


def build_epc_postcode():
    """Assign every home to the nearest postcode centroid and average the EPC scores per postcode."""
    homes, info = read_epc()
    cp = read_codepoint()
    dist, idx = cKDTree(cp[["e", "n"]].to_numpy()).query(homes[["e", "n"]].to_numpy())
    homes["postcode"] = cp["postcode"].to_numpy()[idx]
    homes["dist_m"] = dist
    info.update(postcodes_london=len(cp), dist_median_m=float(np.median(dist)),
                dist_p90_m=float(np.quantile(dist, 0.9)), dist_p99_m=float(np.quantile(dist, 0.99)),
                homes_too_far=int((dist > MAX_DIST_M).sum()))
    homes = homes[homes["dist_m"] <= MAX_DIST_M]

    homes["ac"] = (homes["score"] >= BANDS["C"][0]).astype(float)    # rated C or better

    def summary(h, tag):
        g = h.groupby("postcode")
        return pd.DataFrame({f"n_{tag}": g.size(), f"score_{tag}": g["score"].mean(),
                             f"share_ac_{tag}": g["ac"].mean()})

    pc = summary(homes, "all").join(summary(homes[homes["flat"]], "flat")).join(
        summary(homes[~homes["flat"]], "house"))
    band = pd.cut(homes["score"], [0, 21, 39, 55, 69, 81, 92, 101], right=False, labels=list("GFEDCBA"))
    pc = pc.join(pd.crosstab(homes["postcode"], band).add_prefix("band_"))
    pc = pc.reset_index()
    pc.to_csv(EPC_FILE, index=False)
    EPC_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return pc, info


def add_epc(df):
    """Merge the postcode EPC values to the sales. A flat gets the flats' average, a house the houses' average."""
    if EPC_FILE.exists() and EPC_INFO.exists():
        pc, info = pd.read_csv(EPC_FILE), json.loads(EPC_INFO.read_text(encoding="utf-8"))
    else:
        print("Building data/epc_postcode.csv ...")
        pc, info = build_epc_postcode()
    n = len(df)
    # round, so the values are identical whether they come from the first run or from the cached CSV
    num = [c for c in pc.columns if c.startswith(("score_", "share_ac_"))]
    pc[num] = pc[num].round(6)
    df = df.merge(pc, on="postcode", how="left")
    df["house"] = df[["detached_dum", "semi_d_dum", "terrace_dum"]].sum(axis=1)
    is_flat = df["house"] == 0
    df["epc_n"] = np.where(is_flat, df["n_flat"], df["n_house"])
    df["epc_score"] = np.where(is_flat, df["score_flat"], df["score_house"])
    df["epc_share_ac"] = np.where(is_flat, df["share_ac_flat"], df["share_ac_house"])
    info["sales_any_epc"] = int(df["n_all"].notna().sum())
    df = df[df["epc_n"] >= MIN_HOMES].copy()
    info.update(sales_before=n, sales_after=len(df))
    df["eff_grp"] = pd.cut(df["epc_score"], [g[1] for g in GROUPS.values()] + [101], right=False,
                           labels=list(GROUPS)).astype(str)
    df["lsoa_year"] = df["lsoacode"] + "_" + df["year"].astype(str)
    df["la_year"] = df["oslaua"] + "_" + df["year"].astype(str)
    return df, info


# ---------------------------------------------------------------------------
# Interaction terms
# ---------------------------------------------------------------------------
def with_pm(d, pm="pm10", score="epc_score"):
    """Sample with `pm` as the PM10 measure and `score` as the efficiency measure, plus all interaction terms.

    The efficiency score, the house dummy and the new-build dummy are centred at the sample mean, so the `pm10`
    coefficient is the PM10 effect for a home of average efficiency and the average mix of houses and new builds.
    `eff10` is the efficiency score in units of 10 points (about one rating band).
    """
    d = d.dropna(subset=[pm, score]).copy()
    d["pm10"] = d[pm]
    d["eff10"] = (d[score] - d[score].mean()) / 10
    if score != "epc_score":   # groups follow the efficiency measure that is used
        d["eff_grp"] = pd.cut(d[score], [g[1] for g in GROUPS.values()] + [101], right=False,
                              labels=list(GROUPS)).astype(str)
    d["pm10_x_eff"] = d["pm10"] * d["eff10"]
    for k in ["ac", "eg"]:
        d[f"grp_{k}"] = (d["eff_grp"] == k).astype(int)
        d[f"pm10_x_{k}"] = d["pm10"] * d[f"grp_{k}"]
    d["pm10_x_house"] = d["pm10"] * (d["house"] - d["house"].mean())
    d["pm10_x_new"] = d["pm10"] * (d["newbuild"] - d["newbuild"].mean())
    return d


CONT = ["pm10_x_eff", "eff10"]                              # continuous efficiency
GRP = ["pm10_x_ac", "pm10_x_eg", "grp_ac", "grp_eg"]        # efficiency groups (base: average, D)
PTYPE = ["pm10_x_house", "pm10_x_new"]                      # property-type confounders


def m8(d, extra, cluster=CLUSTER, ptype=True):
    """M8 + efficiency terms (+ PM10 × house and PM10 × new build). Terms that do not vary are dropped."""
    # in a flats-only or houses-only sample PM10 × house is just a multiple of PM10, so it is left out
    pt = [t for t in PTYPE if ptype and not (t == "pm10_x_house" and d["house"].nunique() == 1)]
    xs = ["pm10"] + list(extra) + pt + CONTROLS
    xs = [v for v in xs if d[v].std() > 0]
    return fe_ols(d, xs, time_fe=True, area_fe="lsoacode", cluster=cluster)


def m8_grp_trends(d, extra):
    """Group model where each efficiency group gets its own year-quarter effects."""
    d = d.assign(yearquarter=d["eff_grp"] + "_" + d["yearquarter"].astype(str))
    return m8(d, extra)


def m8_cont_trends(d, extra):
    """Continuous model with efficiency × year dummies (efficient areas may follow a different price trend)."""
    d = d.copy()
    yrs = by_year(d, "eff10")
    return m8(d, list(extra) + yrs)


def m8_la_year(d, extra):
    xs = ["pm10"] + list(extra) + PTYPE + CONTROLS
    return two_fe_ols(d, xs, "lsoacode", "la_year")


def m8_lsoa_year(d, extra):
    """LSOA-by-year fixed effects: compares efficient and inefficient postcodes in the same LSOA and year."""
    xs = varying(d, ["pm10"] + list(extra) + PTYPE + CONTROLS, "lsoa_year")
    return fe_ols(d, xs, time_fe=True, area_fe="lsoa_year")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def has(m, t):
    return m is not None and t in m["res"].index


def cell(m, t):
    if not has(m, t):
        return ""
    c, s = coef(m, t)
    return f"{c * 100:+.3f}{stars(c, s)} ({s * 100:.3f})"


def mtxt(m, t):
    c, s = coef(m, t)
    return f"{c * 100:+.3f} pp ({sig_txt(c, s)}, p {p_fmt(p_normal(c / s))})"


def group_effects(m, d):
    """PM10 effect, its s.e., WTP and mean price for each efficiency group from a group model."""
    out = {}
    for k in GROUPS:
        w = {"pm10": 1} if k == "d" else {"pm10": 1, f"pm10_x_{k}": 1}
        if not all(has(m, t) for t in w):
            continue
        c, s = lincom(m, w)
        p = d.loc[d["eff_grp"] == k, "price"].mean()
        out[k] = {"coef": c, "se": s, "price": p, "wtp": -c * p, "n": int((d["eff_grp"] == k).sum())}
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(panels):
    """panels: list of (title, {group: effect dict})"""
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rows = [(title, k, e[k]) for title, e in panels for k in GROUPS if k in e]
    y = np.arange(len(rows))[::-1].astype(float)
    # leave a gap between the panels
    for i in range(len(rows)):
        y[i] -= 0.6 * [t for t, _ in panels].index(rows[i][0])
    ax.axvline(0, color="#52514e", linewidth=0.8)
    for yi, (title, k, e) in zip(y, rows):
        ax.errorbar(e["coef"] * 100, yi, xerr=1.96 * e["se"] * 100, fmt="o", color=COLORS[k], ecolor="#0b0b0b",
                    elinewidth=1, capsize=3, markersize=7, zorder=3)
        ax.annotate(f"{e['coef'] * 100:+.2f}%", (e["coef"] * 100, yi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color="#0b0b0b")
    ax.set_yticks(y, [f"{title}: {GROUPS[k][0]}" for title, k, _ in rows])
    ax.set_xlabel("% change in price per +1 µg/m³ PM10 (95% CI)")
    ax.set_title("PM10 effect by energy efficiency of the homes in the postcode (v1 model)", loc="left",
                 fontsize=11, color="#0b0b0b")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Efficiency: average EPC rating of the homes of the same type (flat / house) in the postcode. "
             "Each panel is one model. Standard errors clustered by MSOA.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, full, _ = load_data()
    df, _ = add_smoothed_pm10(df, full)
    df, info = add_epc(df)
    d = with_pm(df)
    print(f"Sales with EPC information: {len(d):,}")

    # ===================== Part 1: the efficiency premium (no interaction) =====================
    prem_cont = m8(d, ["eff10"], ptype=False)
    prem_grp = m8(d, ["grp_ac", "grp_eg"], ptype=False)

    # ===================== Part 2: does the PM10 effect depend on efficiency? =====================
    main_c, main_g = m8(d, CONT), m8(d, GRP)
    print(f"main: PM10 x eff = {coef(main_c, 'pm10_x_eff')[0] * 100:+.3f} pp "
          f"(se {coef(main_c, 'pm10_x_eff')[1] * 100:.3f})")

    flats, houses = d[d["house"] == 0], d[d["house"] == 1]
    d_all = with_pm(df, score="score_all")                     # efficiency of all homes in the postcode
    d_all = d_all[d_all["n_all"] >= MIN_HOMES]
    d_ma3 = with_pm(df, pm="pm10_ma3")                         # 3-year average PM10 (v2 measure)
    d_big = d[d["epc_n"] >= 10]                                 # postcodes with many certificates (less noise)

    checks = [  # (label, continuous model, group model)
        ("**Main: M8 + efficiency + PM10 × house + PM10 × new build**", main_c, main_g),
        ("Without PM10 × house and PM10 × new build", m8(d, CONT, ptype=False), m8(d, GRP, ptype=False)),
        ("+ efficiency-specific year effects", m8_cont_trends(d, CONT), m8_grp_trends(d, GRP)),
        ("+ borough-by-year fixed effects", m8_la_year(d, CONT), m8_la_year(d, GRP)),
        ("LSOA-by-year fixed effects", m8_lsoa_year(d, CONT), m8_lsoa_year(d, GRP)),
        ("Standard errors clustered by borough", m8(d, CONT, cluster="oslaua"), m8(d, GRP, cluster="oslaua")),
        ("3-year average PM10 (v2 measure)", m8(d_ma3, CONT), m8(d_ma3, GRP)),
        ("Efficiency of all homes in the postcode (flats and houses together)", m8(d_all, CONT), m8(d_all, GRP)),
        ("Only postcodes with 10+ homes of the same type with an EPC", m8(d_big, CONT), m8(d_big, GRP)),
        ("Flats only", m8(flats, CONT), m8(flats, GRP)),
        ("Houses only", m8(houses, CONT), m8(houses, GRP)),
    ]
    print("robustness done")

    # --- PM10 effect and WTP by group ---
    eff_all = group_effects(main_g, d)
    eff_flat = group_effects(checks[-2][2], flats)
    eff_house = group_effects(checks[-1][2], houses)
    plot([("All homes", eff_all), ("Flats", eff_flat), ("Houses", eff_house)])

    # --- tests ---
    W_eq, _, p_eq = wald(main_g, ["pm10_x_ac", "pm10_x_eg"])           # equal PM10 effect in all three groups
    b_int, se_int = coef(main_c, "pm10_x_eff")
    b_pm, se_pm = coef(main_c, "pm10")
    lo95, hi95 = b_int - 1.96 * se_int, b_int + 1.96 * se_int
    p_one = 0.5 * math.erfc((b_int / se_int) / math.sqrt(2))           # one-sided p for H1: interaction > 0
    hyp_ok = b_int > 0 and p_one < 0.05
    n_pos = sum(coef(c, "pm10_x_eff")[0] > 0 for _, c, _ in checks if has(c, "pm10_x_eff"))
    n_sig_pos = sum(coef(c, "pm10_x_eff")[0] / coef(c, "pm10_x_eff")[1] > 1.96 for _, c, _ in checks
                    if has(c, "pm10_x_eff"))
    n_sig_neg = sum(coef(c, "pm10_x_eff")[0] / coef(c, "pm10_x_eff")[1] < -1.96 for _, c, _ in checks
                    if has(c, "pm10_x_eff"))
    n_checks = sum(has(c, "pm10_x_eff") for _, c, _ in checks)
    ac_minus_d = coef(main_g, "pm10_x_ac")
    ac_t = {label: coef(g, "pm10_x_ac")[0] / coef(g, "pm10_x_ac")[1] for label, _, g in checks if has(g, "pm10_x_ac")}
    n_ac_neg = sum(t < -1.96 for t in ac_t.values())
    ac_not_sig = [label.strip("*").split(":")[0] for label, t in ac_t.items() if abs(t) <= 1.96]

    # ===================== Markdown =====================
    shares = d["eff_grp"].value_counts(normalize=True)
    band_cols = [f"band_{b}" for b in BANDS]
    linked = df.drop_duplicates("postcode")[band_cols].sum()
    band_md = " | ".join(f"{linked[f'band_{b}'] / linked.sum() * 100:.1f}%" for b in BANDS)

    grp_desc = []
    for k, (label, lo, hi) in GROUPS.items():
        g = d[d["eff_grp"] == k]
        grp_desc.append(f"| {label} | {lo} to {hi - 1} | {len(g):,} ({shares.get(k, 0) * 100:.0f}%) | "
                        f"{g['epc_score'].mean():.1f} | {(1 - g['house']).mean() * 100:.0f}% | "
                        f"{g['newbuild'].mean() * 100:.0f}% | {g['pm10'].mean():.1f} | {g['dist_to_cbd'].mean():.1f} | "
                        f"£{g['price'].mean():,.0f} |")
    corr = d[["epc_score", "pm10", "house", "newbuild", "dist_to_cbd"]].corr()["epc_score"]
    within_sd = (d["epc_score"] - d.groupby("lsoacode")["epc_score"].transform("mean")).std()

    rob_md = []
    for label, mc, mg in checks:
        n = mc["n"] if mc is not None else mg["n"]
        rob_md.append(f"| {label} | {cell(mc, 'pm10_x_eff')} | {cell(mg, 'pm10_x_ac')} | {cell(mg, 'pm10_x_eg')} | "
                      f"{n:,} |")

    def eff_rows(title, e):
        rows = []
        for k in GROUPS:
            if k in e:
                x = e[k]
                rows.append(f"| {title} | {GROUPS[k][0]} | {pct(x['coef'])}{stars(x['coef'], x['se'])} | "
                            f"{x['se'] * 100:.3f} | {gbp(x['wtp'])} | £{x['price']:,.0f} | {x['n']:,} |")
        return rows

    eff_md = eff_rows("All homes", eff_all) + eff_rows("Flats", eff_flat) + eff_rows("Houses", eff_house)

    c_pc, s_pc = coef(prem_cont, "eff10")
    c_ac, s_ac = coef(prem_grp, "grp_ac")
    c_eg, s_eg = coef(prem_grp, "grp_eg")

    flat_c = coef(checks[-2][1], "pm10_x_eff")
    house_c = coef(checks[-1][1], "pm10_x_eff")
    noptype_c = coef(checks[1][1], "pm10_x_eff")
    ptype_txt = ("Leaving out the property-type terms makes the interaction more negative, as expected: efficient "
                 "postcodes are full of flats, which carry the larger discount." if noptype_c[0] < b_int else
                 "Leaving out the property-type terms does not make the interaction more negative.")
    if n_ac_neg:
        ac_txt = (f"its PM10 discount is larger than in the average group, the opposite of H1, and significantly so "
                  f"(5%) in {n_ac_neg} of {len(ac_t)} specifications. It is not robust: it is not significant at 5% "
                  f"in {len(ac_not_sig)} of them ({'; '.join(ac_not_sig)}). Efficient postcodes are mostly new-build "
                  f"flat developments, so the difference may come from these developments rather than from "
                  f"efficiency itself. We read it as a weak hint, not as a finding.")
    else:
        ac_txt = "its PM10 effect is not significantly different from the average group in any specification."
    max_share = max(hi95, 0) / abs(b_pm)          # largest share of the discount one band could remove
    rule_out = ("A large protective effect (half of the discount or more gone per band) is outside the interval."
                if max_share < 0.5 else
                "The interval is wide: a large protective effect cannot be ruled out.")

    if hyp_ok:
        verdict = ("**Supported.** The PM10 discount is significantly smaller where homes are more energy efficient "
                   "(one-sided test).")
    elif b_int > 0:
        verdict = ("**Not supported.** The interaction has the expected sign but is not significant, so the data "
                   "cannot tell it apart from zero.")
    else:
        verdict = ("**Not supported.** The interaction does not have the expected sign: if anything, the PM10 "
                   "discount is slightly *larger* where homes are more efficient, but this is not significant.")
        if b_int / se_int < -1.96:
            verdict = ("**Rejected.** The interaction is significantly negative: the PM10 discount is *larger* where "
                       "homes are more efficient, the opposite of the hypothesis.")

    md = f"""# Model v1 + Energy Efficiency: Are Efficient Homes Less Affected by Outdoor Air?

Generated by `model_v1_energy_efficiency.py`. All numbers in this file are produced by the script.
Outcome: log sale price. Coefficients ×100 give the % change in price.

## Question and hypothesis

Energy-efficient homes are better insulated, more often double glazed and less draughty. They exchange less air
with the outside, so the air inside them should depend less on outdoor PM10. If buyers know or feel this, an
efficient home should lose **less** value when the PM10 around it rises.

**Hypothesis (H1):** the PM10 effect on prices is smaller (less negative) for energy-efficient homes, i.e. the
interaction **PM10 × energy efficiency is positive**. We fix the test before looking at the results: the main
specification below, with a one-sided test at 5%, and the robustness checks in the table.

## Data

| Data | What we use | Level | Year | Source |
|---|---|---|---|---|
| House sales, PM10, controls | As in the v1 model (M8) | Each sale | 2005-2014 | Course data set `land_registry_pollution.dta` |
| **Energy Performance Certificates (new)** | Current energy rating band, A (best) to G (worst), and the address point of the home | Each home (UPRN) | Certificates lodged since 2008, all years (the copy has no dates) | Ministry of Housing, Communities and Local Government, EPC register, copy in geoparquet by [Addresscloud on Source Cooperative](https://source.coop/addresscloud/epc), Open Government Licence |
| **Postcode centroids (new)** | Location of every postcode, to place the homes in postcodes | Postcode | Current release | [OS Code-Point Open](https://osdatahub.os.uk/downloads/open/CodePointOpen), Open Government Licence |

**What the EPC rating is.** An EPC is needed whenever a home is built, sold or let. An assessor records the walls,
roof, floor, windows, heating and hot water, and software (SAP) turns this into an energy efficiency score from
1 to 100+, shown as a band:

| Band | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| SAP score | 92+ | 81-91 | 69-80 | 55-68 | 39-54 | 21-38 | 1-20 |
| Share of linked homes | {band_md} |

The copy we use has the band only, so each band gets the middle of its score range (A = 96, ..., G = 10.5).
The full EPC register also has the qualitative descriptions behind the rating (glazing, wall insulation,
ventilation, floor level), but it needs a GOV.UK One Login account; see *Next steps*.

### From certificates to sales

Our sales data have a postcode but no house number, so we cannot find the certificate of the home that was sold.
We describe the **homes of the same type in the same postcode** instead (a postcode has about 15 addresses):

1. **Homes:** {info['certificates']:,} certificates in the {info['boroughs']} London boroughs belong to
   {info['homes']:,} homes; {info['homes_multi_cert']:,} homes have more than one certificate and get the mean of
   their scores.
2. **Flat or house:** a home that shares its address point with other homes is a flat in a multi-unit building
   ({info['flat_share'] * 100:.0f}% of homes); the others are counted as houses.
3. **Postcode:** each home is assigned to the nearest postcode centroid (median distance
   {info['dist_median_m']:.0f} m, 99% within {info['dist_p99_m']:.0f} m). {info['homes_too_far']:,} homes more than
   {MAX_DIST_M} m from any postcode are dropped (geocoding errors).
4. **Sale:** a flat gets the average score of the flats in its postcode, a house the average score of the houses.
   We keep a sale if at least {MIN_HOMES} homes of its type in its postcode have an EPC.

{info['sales_any_epc']:,} of the {info['sales_before']:,} sales in the v1 sample are in a postcode with any EPC, and
**{len(d):,} sales** ({len(d) / info['sales_before'] * 100:.0f}%) pass step 4. The efficiency score varies a lot
within LSOAs (standard deviation {within_sd:.1f} points within LSOAs, {d['epc_score'].std():.1f} overall), which is
what the LSOA fixed-effects model needs.

### Efficiency groups

| Group | Average score | Sales | Mean score | Flats | New builds | PM10 (µg/m³) | Distance to centre (km) | Mean price |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(grp_desc)}

Correlation of the efficiency score with: PM10 {corr['pm10']:.2f}, house {corr['house']:.2f},
new build {corr['newbuild']:.2f}, distance to the centre {corr['dist_to_cbd']:.2f}.

**The main confounder.** Efficient postcodes have many more flats and new builds. Flats lose more value from
PM10 than houses (`model_v2.md`), so a simple PM10 × efficiency term would partly pick up PM10 × flat. The main
specification therefore also includes **PM10 × house** and **PM10 × new build**, and the checks estimate flats
and houses separately.

## Part 1: Do efficient homes sell for more?

Within an LSOA and quarter (M8 controls, no PM10 interaction):

| Term | Effect on price | Std. error |
|---|---|---|
| +10 points of efficiency score (about one band) | {pct(c_pc)}{stars(c_pc, s_pc)} | {s_pc * 100:.2f} |
| Efficient (A-C) vs average (D) postcode | {pct(c_ac)}{stars(c_ac, s_ac)} | {s_ac * 100:.2f} |
| Inefficient (E-G) vs average (D) postcode | {pct(c_eg)}{stars(c_eg, s_eg)} | {s_eg * 100:.2f} |

This is a by-product, not the question: the postcode score also stands for the age and type of the buildings,
so it is not a clean "green premium". Compared with average (D) postcodes, both ends can sell for more for
different reasons: inefficient postcodes are mostly older (often period) houses, efficient postcodes mostly
new-build flats (see the group table above).

## Part 2: Does the PM10 effect depend on efficiency?

ln(price) = β · PM10 + γ · PM10 × efficiency + δ · efficiency + θ₁ · PM10 × house + θ₂ · PM10 × new build
+ M8 controls + LSOA fixed effects + year-quarter fixed effects + error

Efficiency, house and new build are centred, so β is the PM10 effect for a home of average efficiency and the
average mix of property types. **γ > 0 means a smaller discount for efficient homes (H1).** PM10 changes over time
within an LSOA, so γ asks whether prices in efficient postcodes react less to the *same change* in PM10 than prices
in inefficient postcodes of the same LSOA.

### Main result

- **PM10 effect at average efficiency:** {pct(b_pm)}{stars(b_pm, se_pm)} per µg/m³ (standard error {se_pm * 100:.3f} pp).
- **PM10 × efficiency:** {mtxt(main_c, 'pm10_x_eff')} per +10 score points.
  95% confidence interval [{lo95 * 100:+.3f}, {hi95 * 100:+.3f}] pp. One-sided p-value for H1 (γ > 0): {p_fmt(p_one)}.
- **Groups:** efficient (A-C) minus average (D): {mtxt(main_g, 'pm10_x_ac')};
  inefficient (E-G) minus average (D): {mtxt(main_g, 'pm10_x_eg')}.
  Joint test that the PM10 effect is the same in all three groups: W = {W_eq:.2f}, p = {p_fmt(p_eq)}.

**How large an effect can we rule out?** The upper end of the confidence interval is {hi95 * 100:+.3f} pp per band.
Compared with the PM10 effect of {pct(b_pm)}, a home one band more efficient could at most have a
{max_share * 100:.0f}% smaller discount. {rule_out}

### Robustness

| Specification | PM10 × efficiency (per +10 points) | PM10 × efficient (A-C) − average (D) | PM10 × inefficient (E-G) − average (D) | Sales |
|---|---|---|---|---|
{chr(10).join(rob_md)}

Coefficient ×100, in percentage points of the PM10 effect (standard error ×100). \\* p < 0.10, \\*\\* p < 0.05,
\\*\\*\\* p < 0.01. Two-sided stars. Empty cell: not estimable in that specification.

- **Without PM10 × house and PM10 × new build:** {noptype_c[0] * 100:+.3f} pp instead of {b_int * 100:+.3f} pp.
  {ptype_txt}
- **Efficiency-specific year effects:** efficient and inefficient postcodes may follow different price trends
  (for example new-build areas); here each group, or each efficiency level, gets its own year effects.
- **Borough-by-year fixed effects:** remove any shock that hits a whole borough in a given year.
- **LSOA-by-year fixed effects:** compare efficient and inefficient postcodes in the same LSOA in the same year, so
  any local shock cancels out. PM10 itself is absorbed (it hardly varies within an LSOA-year); the interaction is
  still identified.
- **Flats only / houses only:** flats {flat_c[0] * 100:+.3f} pp ({sig_txt(*flat_c)}), houses
  {house_c[0] * 100:+.3f} pp ({sig_txt(*house_c)}). Within one property type the flat-house confounder is gone.

Across the {n_checks} specifications the continuous interaction is positive in {n_pos}, significantly positive
(5%) in {n_sig_pos} and significantly negative in {n_sig_neg}.

### PM10 effect for each efficiency group

| Homes | Efficiency group | PM10 effect | Std. error (pp) | WTP per µg/m³ | Mean price | Sales |
|---|---|---|---|---|---|---|
{chr(10).join(eff_md)}

WTP = −effect × mean price of the group. "All homes" is the main group model; flats and houses are separate models.

![PM10 effect by energy efficiency](../figures/energy_efficiency.png)

## Verdict

{verdict}

- PM10 × efficiency is {b_int * 100:+.3f} pp per band (p {p_fmt(p_normal(b_int / se_int))}, two-sided), and the
  efficient group (A-C) differs from the average group by {ac_minus_d[0] * 100:+.3f} pp ({sig_txt(*ac_minus_d)}).
- Robustness: significantly positive (as H1 predicts) in {n_sig_pos} of {n_checks} specifications, significantly
  negative in {n_sig_neg}.
- **Efficient group (A-C):** {ac_txt}
- Size: one band more efficiency removes at most {max_share * 100:.0f}% of the PM10 discount (upper end of the
  95% interval). {rule_out}

## Why we might not find an effect even if efficiency protects indoor air

- **Postcode average, not the home.** We do not know the rating of the home that was sold, only the average of the
  homes of the same type in its postcode. This measurement error pulls γ towards zero.
- **Timing.** The certificates were lodged between 2008 and today, while the sales are from 2005-2014. Many homes
  were improved (new windows, insulation, boilers) after they were sold, so today's rating overstates the efficiency
  at the time of sale. Homes keep their *relative* position fairly well (building type and age matter most).
- **Energy is not air.** The SAP score measures energy use. A new condensing boiler raises it a lot but does nothing
  for the air inside; double glazing and insulation help both. Airtightness is not part of the rating for most
  existing homes.
- **Buyers.** Even if efficient homes do have cleaner indoor air, buyers may not know it, and the hedonic price only
  reflects what buyers believe.
- **PM10 from monitoring sites.** As in all our models, PM10 is the yearly mean at the nearest site, measured with
  error; interactions with a noisy PM10 are biased towards zero as well.
- **Not causal.** Like M8, these are associations conditional on the controls and fixed effects.

## Next steps

The full EPC register ([get-energy-performance-data.communities.gov.uk](https://get-energy-performance-data.communities.gov.uk/),
free, needs a GOV.UK One Login account) has the qualitative parts of the certificate for each home, with postcode,
property type and inspection date: **glazing** (single / double / triple), **wall and roof insulation**,
**mechanical ventilation**, **floor level** and **construction age band**. With it we could
(1) use the certificate closest to the sale date, (2) match on property type and floor, and (3) test the mechanism
directly with PM10 × double glazing and PM10 × mechanical ventilation, and PM10 × floor level for flats, which is
the open question from `model_v2_roads.md`.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE} and {FIG_FILE}")


if __name__ == "__main__":
    main()
