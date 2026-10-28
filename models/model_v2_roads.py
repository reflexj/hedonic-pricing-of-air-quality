"""
HE3027 Group Project: model v2 with major roads and road noise.

Most PM10 and NOx in London comes from road traffic, and main roads are also noisy. Our PM10 is the yearly mean
at the nearest monitoring site, so it cannot see whether a home sits right next to a main road. This script adds,
for every postcode,
  - the distance to the nearest motorway or A road        (OS Open Roads)
  - road noise, Lden in 5 dB bands, 2012                   (Defra strategic noise mapping, round 2)
and asks three questions:
  1. Do main roads and road noise lower prices (within an LSOA)?
  2. Do they change the PM10 effect of model v2?
  3. Does the larger PM10 discount for flats (model_v2.md) come from flats on busy roads?

Writes model_v2_roads.md and figures/v2_roads.png. The road and noise values per postcode are cached in
data/road_features.csv, which is built on the first run (delete it to rebuild).

Run from the repository root:
    python models/model_v2_roads.py

Needs: pandas, numpy, matplotlib, scipy, shapely, pyproj, pyshp
Data:  data/land_registry_pollution.dta, data/gla_lsoa_atlas/lsoa-data.csv,
       data/os_open_roads/oproad_essh_gb.zip
           OS Open Roads, ESRI Shapefile, GB: https://osdatahub.os.uk/downloads/open/OpenRoads
       data/defra_road_noise/Road_Lden_London.shp (+ .dbf, .shx, .prj)
           Road_Lden_London.zip, unzipped: https://data.london.gov.uk/dataset/noise-pollution-in-london-2zwnk
"""

import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
import shapely
from pyproj import Transformer
from scipy import stats

from model_v1 import CLUSTER, CODE_DIR, MODEL_DIR, DATA_DIR, fe_ols, load_data, pct, stars, two_fe_ols
from model_v1_income_groups import p_fmt, p_normal
from model_v1_pm10_smoothing import add_smoothed_pm10
from model_v2 import CONTROLS_V2, add_lsoa_atlas, add_v2_terms, cell, coef, m8v2, sig_txt, varying, with_pm

MD_FILE = MODEL_DIR / "model_v2_roads.md"
FIG_DIR = CODE_DIR / "figures"
FIG_FILE = FIG_DIR / "v2_roads.png"

ROADS_ZIP = DATA_DIR / "os_open_roads" / "oproad_essh_gb.zip"
NOISE_SHP = DATA_DIR / "defra_road_noise" / "Road_Lden_London"
FEATURES = DATA_DIR / "road_features.csv"

ROAD_TILES = ["TQ", "TL"]          # 100 km grid squares: London lies in TQ, its northern edge reaches into TL
MAJOR = {"Motorway", "A Road"}     # OS Open Roads `class`
NEAR_M = 100                       # "near a main road": within 100 m

ROAD_BANDS = [  # name, label, from (m), to (m); base group: 400 m or more
    ("road_0_50", "0-50 m", 0, 50),
    ("road_50_100", "50-100 m", 50, 100),
    ("road_100_200", "100-200 m", 100, 200),
    ("road_200_400", "200-400 m", 200, 400),
]
NOISE_BANDS = [  # name, label, lower bound of the Defra band (dB); base group: below 55 dB (not mapped)
    ("noise_55", "55-59.9 dB", 55),
    ("noise_60", "60-64.9 dB", 60),
    ("noise_65", "65-69.9 dB", 65),
    ("noise_70", "70-74.9 dB", 70),
    ("noise_75", "75 dB or more", 75),
]
ROAD_VARS = [b[0] for b in ROAD_BANDS]
NOISE_VARS = [b[0] for b in NOISE_BANDS]
GROUPS = [  # PM10 effect by property type and road proximity
    ("pm10_flat_near", "Flats within 100 m of a main road"),
    ("pm10_flat_far", "Flats further away"),
    ("pm10_house_near", "Houses within 100 m of a main road"),
    ("pm10_house_far", "Houses further away"),
]


# ---------------------------------------------------------------------------
# Road and noise values per postcode (built once, then read from the cache)
# ---------------------------------------------------------------------------
def read_major_roads():
    """Motorways and A roads from the OS Open Roads RoadLink layers of the London grid squares."""
    z = zipfile.ZipFile(ROADS_ZIP)
    lines, primary = [], []
    for tile in ROAD_TILES:
        base = next(n[:-4] for n in z.namelist() if n.endswith(f"{tile}_RoadLink.shp"))
        r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), shx=io.BytesIO(z.read(base + ".shx")),
                             dbf=io.BytesIO(z.read(base + ".dbf")))
        fields = [f[0] for f in r.fields[1:]]
        i_class, i_primary = fields.index("class"), fields.index("primary")
        for sr in r.iterShapeRecords():
            if sr.record[i_class] in MAJOR:
                lines.append(shapely.linestrings(sr.shape.points))
                primary.append(sr.record[i_class] == "Motorway" or str(sr.record[i_primary]).lower() == "true")
    return np.array(lines), np.array(primary)


def read_noise():
    r = shapefile.Reader(str(NOISE_SHP))
    polys = [shapely.geometry.shape(s.__geo_interface__) for s in r.iterShapes()]
    level = np.array([int(float(rec[0].replace(">=", "").split("-")[0])) for rec in r.iterRecords()])
    return np.array(polys), level


def build_features(df):
    pc = df.groupby("postcode", as_index=False)[["lat", "lon"]].first()
    e, n = Transformer.from_crs(4326, 27700, always_xy=True).transform(pc["lon"].to_numpy(), pc["lat"].to_numpy())
    pts = shapely.points(e, n)
    pc["easting"], pc["northing"] = e, n

    lines, primary = read_major_roads()
    for name, sel in [("dist_major_m", np.ones(len(lines), bool)), ("dist_primary_m", primary)]:
        tree = shapely.STRtree(lines[sel])
        _, dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
        pc[name] = dist
    print(f"  {len(lines):,} main road links, {primary.sum():,} of them motorways or primary routes")

    polys, level = read_noise()
    pi, gi = shapely.STRtree(polys).query(pts, predicate="intersects")
    noise = pd.Series(level[gi]).groupby(pi).max()            # loudest band the postcode falls into
    pc["noise_lden"] = 0
    pc.loc[noise.index, "noise_lden"] = noise.to_numpy()
    pc.drop(columns=["lat", "lon"]).to_csv(FEATURES, index=False)
    return pc


def add_road_features(df):
    if FEATURES.exists():
        pc = pd.read_csv(FEATURES)
    else:
        print("Building data/road_features.csv ...")
        pc = build_features(df)
    df = df.merge(pc[["postcode", "dist_major_m", "dist_primary_m", "noise_lden"]], on="postcode", how="left")
    for name, _, lo, hi in ROAD_BANDS:
        df[name] = df["dist_major_m"].between(lo, hi, inclusive="left").astype(int)
    for name, _, lo in NOISE_BANDS:
        df[name] = (df["noise_lden"] == lo).astype(int)
    df["near"] = (df["dist_major_m"] < NEAR_M).astype(int)
    df["loud"] = (df["noise_lden"] >= 65).astype(int)
    df["primary_near"] = (df["dist_primary_m"] < NEAR_M).astype(int)
    return df


def add_road_interactions(d):
    d["pm10_x_near"] = d["pm10"] * d["near"]
    d["pm10_x_loud"] = d["pm10"] * d["loud"]
    flat = 1 - d["house"]
    for name, g in zip([g[0] for g in GROUPS], [flat * d["near"], flat * (1 - d["near"]),
                                                 d["house"] * d["near"], d["house"] * (1 - d["near"])]):
        d[name] = d["pm10"] * g
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wald_chi2(m, terms):
    """Clustered Wald test that all `terms` are zero (chi² form)."""
    b = m["res"].loc[terms, "coef"].to_numpy()
    V = m["V"].loc[terms, terms].to_numpy()
    W = float(b @ np.linalg.solve(V, b))
    return W, len(terms), stats.chi2.sf(W, len(terms))


def price_pct(b):
    """Coefficient of a dummy in a log-price model as a % price difference."""
    return f"{(np.exp(b) - 1) * 100:+.1f}%"


def mtxt(m, t):
    c, s = coef(m, t)
    return f"{c * 100:+.3f} pp ({sig_txt(c, s)}, p {p_fmt(p_normal(c / s))})"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(main, groups):
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.1, 1]})

    # left: price difference by distance band and noise band (model R3); the 75 dB+ band is left out because
    # its wide interval would squash the others (its value is given in the footnote)
    rows = [("Distance to main road", lab, v) for v, lab, *_ in ROAD_BANDS] + \
           [("Road noise (Lden)", lab, v) for v, lab, _ in NOISE_BANDS if v != "noise_75"]
    y = np.arange(len(rows))[::-1]
    colors = {"Distance to main road": "#2a78d6", "Road noise (Lden)": "#eb6834"}
    a1.axvline(0, color="#52514e", linewidth=0.8)
    for yi, (g, lab, v) in zip(y, rows):
        c, s = coef(main, v)
        a1.errorbar(c * 100, yi, xerr=1.96 * s * 100, fmt="o", color=colors[g], ecolor="#0b0b0b", elinewidth=1,
                    capsize=3, markersize=7, zorder=3)
        a1.annotate(f"{c * 100:+.1f}", (c * 100, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color="#0b0b0b")
    a1.set_yticks(y, [r[1] for r in rows])
    a1.set_xlabel("Price difference, ×100 log points (95% CI)")
    a1.set_title("Main roads and noise (base: 400 m+, below 55 dB)", loc="left", fontsize=11, color="#0b0b0b")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=g) for g, c in colors.items()]
    fig.legend(handles=handles, frameon=False, loc="lower left", ncol=2, fontsize=8, bbox_to_anchor=(0.01, 0.04))

    # right: PM10 effect by property type and road proximity
    y = np.arange(len(GROUPS))[::-1]
    a2.axvline(0, color="#52514e", linewidth=0.8)
    for yi, (v, lab) in zip(y, GROUPS):
        c, s = coef(groups, v)
        col = "#2a78d6" if "flat" in v else "#1f9e74"
        a2.errorbar(c * 100, yi, xerr=1.96 * s * 100, fmt="o", color=col, ecolor="#0b0b0b", elinewidth=1,
                    capsize=3, markersize=7, zorder=3)
        a2.annotate(f"{c * 100:+.2f}%", (c * 100, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color="#0b0b0b")
    a2.set_yticks(y, [g[1].replace(" within", "\nwithin") for g in GROUPS])
    a2.set_xlabel("% change in price per +1 µg/m³ PM10, 3-year average (95% CI)")
    a2.set_title("PM10 effect by property type and road", loc="left", fontsize=11, color="#0b0b0b")

    for ax in (a1, a2):
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    c75, s75 = coef(main, "noise_75")
    fig.text(0.01, 0.01, "Model v2 (LSOA and year-quarter fixed effects) with all road and noise bands. "
             f"Not shown: 75 dB or more, {c75 * 100:+.1f} (s.e. {s75 * 100:.1f}), very few sales. "
             "Standard errors clustered by MSOA.", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, full, info = load_data()
    df, _ = add_smoothed_pm10(df, full)
    df, _ = add_lsoa_atlas(df)
    add_v2_terms(df)
    df = add_road_features(df)
    n_missing = int(df["dist_major_m"].isna().sum())
    df = df.dropna(subset=["dist_major_m"])

    d = add_road_interactions(with_pm(df, "pm10_ma3"))           # v2 sample: 3-year average PM10
    dy = add_road_interactions(with_pm(df, "pm10"))              # yearly PM10, full sample
    ROADS = ROAD_VARS + NOISE_VARS

    # ===================== Part 1: main effects =====================
    m_v2 = m8v2(d)
    m_road = m8v2(d, controls=CONTROLS_V2 + ROAD_VARS)
    m_noise = m8v2(d, controls=CONTROLS_V2 + NOISE_VARS)
    m_both = m8v2(d, controls=CONTROLS_V2 + ROADS)
    d_alt = d.assign(**{v: d["dist_primary_m"].between(lo, hi, inclusive="left").astype(int)
                        for v, _, lo, hi in ROAD_BANDS})
    m_primary = m8v2(d_alt, controls=CONTROLS_V2 + ROADS)
    xs = varying(d, ROADS + CONTROLS_V2, "lsoa_year")
    m_ly = fe_ols(d, xs, time_fe=True, area_fe="lsoa_year")      # same LSOA, same year
    print("main effects done")

    # ===================== Part 2: interactions =====================
    ctrl = CONTROLS_V2 + ROADS
    i_near = m8v2(d, ["pm10_x_near"], controls=ctrl)
    i_loud = m8v2(d, ["pm10_x_loud"], controls=ctrl)
    i_both = m8v2(d, ["pm10_x_near", "pm10_x_loud"], controls=ctrl)
    rob = {
        "Yearly PM10 instead of the 3-year average": {t: m8v2(dy, [t], controls=ctrl)
                                                       for t in ["pm10_x_near", "pm10_x_loud"]},
        "+ borough-by-year fixed effects": {t: two_fe_ols(d, ["pm10", t] + ctrl, "lsoacode", "la_year")
                                            for t in ["pm10_x_near", "pm10_x_loud"]},
        "LSOA-by-year fixed effects": {t: fe_ols(d, varying(d, [t] + ctrl, "lsoa_year"), time_fe=True,
                                                 area_fe="lsoa_year")
                                       for t in ["pm10_x_near", "pm10_x_loud"]},
        "Standard errors clustered by borough": {t: m8v2(d, [t], controls=ctrl, cluster="oslaua")
                                                 for t in ["pm10_x_near", "pm10_x_loud"]},
    }
    print("road interactions done")

    # does the flat discount survive the road controls?
    h_v2 = m8v2(d, ["pm10_x_house"])                                           # as in model_v2.md
    h_ctrl = m8v2(d, ["pm10_x_house"], controls=ctrl)                          # + road and noise bands
    h_full = m8v2(d, ["pm10_x_house", "pm10_x_near", "pm10_x_loud"], controls=ctrl)
    g = fe_ols(d, [v for v, _ in GROUPS] + ctrl, time_fe=True, area_fe="lsoacode")
    w_dd = {"pm10_flat_near": 1, "pm10_flat_far": -1, "pm10_house_near": -1, "pm10_house_far": 1}
    b_dd = sum(w * g["res"].at[k, "coef"] for k, w in w_dd.items())
    a = np.array(list(w_dd.values()), dtype=float)
    se_dd = float(np.sqrt(a @ g["V"].loc[list(w_dd), list(w_dd)].to_numpy() @ a))
    plot(m_both, g)

    # ===================== Descriptives =====================
    by_type = d.groupby("house").agg(sales=("price", "size"), near=("near", "mean"), loud=("loud", "mean"),
                                     dist=("dist_major_m", "median"))
    band_share = pd.crosstab(pd.cut(d["dist_major_m"], [0, 50, 100, 200, 400, np.inf], right=False,
                                    labels=["0-50 m", "50-100 m", "100-200 m", "200-400 m", "400 m+"]),
                             d["noise_lden"].map({0: "<55", 55: "55-59", 60: "60-64", 65: "65-69", 70: "70-74",
                                                  75: "75+"}), normalize="index")
    band_share = band_share[[c for c in ["<55", "55-59", "60-64", "65-69", "70-74", "75+"] if c in band_share]]
    n_sales = d.groupby(pd.cut(d["dist_major_m"], [0, 50, 100, 200, 400, np.inf], right=False))["price"].size()
    noise_n = d["noise_lden"].value_counts()
    loudest = d[d["noise_75"] == 1]
    within_sd = (d["dist_major_m"] - d.groupby("lsoacode")["dist_major_m"].transform("mean")).std()
    corr = d[["dist_major_m", "noise_lden", "pm10"]].corr()

    # ===================== Markdown =====================
    mean_price = d["price"].mean()

    def main_row(label, v):
        cells = []
        for m in [m_road, m_noise, m_both, m_primary, m_ly]:
            if v in m["res"].index:
                c, s = coef(m, v)
                cells.append(f"{c * 100:+.2f}{stars(c, s)} ({s * 100:.2f})")
            else:
                cells.append("")
        return f"| {label} | " + " | ".join(cells) + " |"

    main_rows = [main_row(f"Main road {lab}", v) for v, lab, *_ in ROAD_BANDS] + \
                [main_row(f"Noise {lab}", v) for v, lab, _ in NOISE_BANDS]
    pm_cells = []
    for m in [m_road, m_noise, m_both, m_primary]:
        c, s = coef(m, "pm10")
        pm_cells.append(f"{pct(c)}{stars(c, s)} ({s * 100:.3f})")
    W_r, q_r, p_r = wald_chi2(m_both, ROAD_VARS)
    W_n, q_n, p_n = wald_chi2(m_both, NOISE_VARS)
    W_r1, _, p_r1 = wald_chi2(m_road, ROAD_VARS)
    W_n1, _, p_n1 = wald_chi2(m_noise, NOISE_VARS)

    def irow(label, fits):
        cells = []
        for t in ["pm10_x_near", "pm10_x_loud"]:
            m = fits[t] if isinstance(fits, dict) and t in fits else fits
            c, s = coef(m, t)
            cells.append(cell(c, s))
        return f"| {label} | " + " | ".join(cells) + " |"

    irows = [irow("**One interaction at a time (main)**", {"pm10_x_near": i_near, "pm10_x_loud": i_loud}),
             irow("Both together", i_both)] + [irow(k, v) for k, v in rob.items()]

    grp_rows = []
    for v, lab in GROUPS:
        c, s = coef(g, v)
        grp_sales = int((d[v] != 0).sum())
        p = d.loc[d[v] != 0, "price"].mean()
        grp_rows.append(f"| {lab} | {pct(c)}{stars(c, s)} | {s * 100:.3f} | £{-c * p:,.0f} | {grp_sales:,} |")

    band_md = ["| Distance to main road | " + " | ".join(f"{c} dB" for c in band_share.columns) + " | Sales |",
               "|---|" + "---|" * (len(band_share.columns) + 1)]
    for (idx, row), n in zip(band_share.iterrows(), n_sales):
        band_md.append(f"| {idx} | " + " | ".join(f"{x * 100:.0f}%" for x in row) + f" | {n:,} |")

    c_near, s_near = coef(i_near, "pm10_x_near")
    c_loud, s_loud = coef(i_loud, "pm10_x_loud")
    c_h0, s_h0 = coef(h_v2, "pm10_x_house")
    c_h1, s_h1 = coef(h_ctrl, "pm10_x_house")
    c_h2, s_h2 = coef(h_full, "pm10_x_house")
    b_v2, s_v2 = coef(m_v2, "pm10")
    b_both, s_both = coef(m_both, "pm10")

    md = f"""# Model v2 with Major Roads and Road Noise

Generated by `model_v2_roads.py`. Builds on model v2 (`model_v2.md`): log price on the 3-year average of PM10,
property, log-quadratic distance, amenity and socioeconomic controls (borough pay cubic), LSOA and year-quarter
fixed effects, standard errors clustered by MSOA. All numbers in this file are produced by the script.

## Why roads

Road traffic is the main local source of PM10 and NOx in London, and main roads are also noisy. Our PM10 is the
yearly mean at the nearest of about 60 to 74 monitoring sites, so two homes in the same LSOA get the same PM10 even
if one is on a main road and the other on a quiet side street. Distance to a main road and road noise measure
exactly that local difference. They help in two ways:

- **as controls**: they vary *within* an LSOA (from postcode to postcode), so the LSOA fixed effects do not absorb
  them, and noise lets us separate the noise part of living next to a main road from the rest;
- **as moderators**: PM10 × near a main road asks whether the PM10 discount is larger right next to traffic, and
  it tests the explanation for the flat result in `model_v2.md` (flats are more often on main roads).

## Data

| Data | What we use | Level | Year | Source |
|---|---|---|---|---|
| OS Open Roads | Road links classed as motorway or A road ("main road"); motorways and primary routes as a stricter version | Road network | Current release (major roads in London changed little since 2005-2014) | [Ordnance Survey, OS Open Roads](https://osdatahub.os.uk/downloads/open/OpenRoads), Open Government Licence |
| Road noise, Lden | Modelled 24-hour noise level with evening and night weighting, in 5 dB bands from 55 dB; below 55 dB is not mapped | 10 m grid at 4 m height, as polygons | 2012 (round 2) | [Defra strategic noise mapping, via the London Datastore](https://data.london.gov.uk/dataset/noise-pollution-in-london-2zwnk), Open Government Licence |

Both are matched to the **postcode centroid** of each sale (the data have no exact address): the distance to the
nearest main road link, and the noise band that the centroid falls into. A London postcode covers about 15
addresses, so the location is accurate to a few tens of metres. This adds noise to the 0-50 m band in particular.
{n_missing:,} sales could not be matched. Sample: the v2 sample, {len(d):,} sales.

## Descriptives

| | Sales | Within {NEAR_M} m of a main road | Noise 65 dB or more | Median distance to a main road |
|---|---|---|---|---|
| Flats | {by_type.at[0, 'sales']:,} | {by_type.at[0, 'near'] * 100:.0f}% | {by_type.at[0, 'loud'] * 100:.0f}% | {by_type.at[0, 'dist']:.0f} m |
| Houses | {by_type.at[1, 'sales']:,} | {by_type.at[1, 'near'] * 100:.0f}% | {by_type.at[1, 'loud'] * 100:.0f}% | {by_type.at[1, 'dist']:.0f} m |

**Noise band by distance to a main road** (row shares):

{chr(10).join(band_md)}

Sales per noise band: below 55 dB {noise_n.get(0, 0):,}, 55-59.9 dB {noise_n.get(55, 0):,}, 60-64.9 dB
{noise_n.get(60, 0):,}, 65-69.9 dB {noise_n.get(65, 0):,}, 70-74.9 dB {noise_n.get(70, 0):,}, 75 dB or more
{noise_n.get(75, 0):,} (in {loudest['postcode'].nunique()} postcodes and {loudest['lsoacode'].nunique()} LSOAs).
The loudest band is small, so its coefficient rests on a few streets.

Correlations over sales: distance and noise level {corr.at['dist_major_m', 'noise_lden']:.2f}, distance and PM10
{corr.at['dist_major_m', 'pm10']:.2f}, noise and PM10 {corr.at['noise_lden', 'pm10']:.2f}. The distance to a main road
varies within LSOAs (standard deviation {within_sd:.0f} m after removing LSOA means), which is what identifies
its effect with LSOA fixed effects.

## Part 1: Do main roads and noise lower prices?

Coefficients ×100 (≈ % price difference for small values) with standard errors ×100. Base groups: 400 m or more
from a main road, noise below 55 dB.

| | Roads | Noise | Both | Both, primary roads only | Both, LSOA-by-year FE |
|---|---|---|---|---|---|
{chr(10).join(main_rows)}
| PM10 (3-year average) | {' | '.join(pm_cells)} | absorbed |
| Sales | {m_road['n']:,} | {m_noise['n']:,} | {m_both['n']:,} | {m_primary['n']:,} | {m_ly['n']:,} |

\\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01. "Primary roads only": distance bands measured to motorways and
primary A roads (the strategic network) instead of all A roads. LSOA-by-year FE: compares homes sold in the same
LSOA in the same year; PM10 does not vary within an LSOA-year, so it drops out.

Joint tests (clustered Wald, χ²):

| Test | χ² | df | p |
|---|---|---|---|
| All road bands = 0, roads only | {W_r1:.1f} | {q_r} | {p_fmt(p_r1)} |
| All noise bands = 0, noise only | {W_n1:.1f} | {q_n} | {p_fmt(p_n1)} |
| All road bands = 0, given noise | {W_r:.1f} | {q_r} | {p_fmt(p_r)} |
| All noise bands = 0, given roads | {W_n:.1f} | {q_n} | {p_fmt(p_n)} |

**The PM10 effect with road and noise controls:** {pct(b_both)} (s.e. {s_both * 100:.3f} pp), compared with
{pct(b_v2)} in model v2 on the same sales. Roads and noise are fixed over time, while the PM10 effect with LSOA
fixed effects comes from changes over time, so {'they hardly change it' if abs(b_both - b_v2) < 0.5 * s_v2 else 'they change it noticeably'}.

![Main roads, noise and the PM10 effect](../figures/v2_roads.png)

## Part 2: Is the PM10 discount larger next to a main road?

Model v2 with all road and noise bands as controls, plus PM10 × near (within {NEAR_M} m of a main road) and
PM10 × loud (noise 65 dB or more). Coefficients ×100 = difference in the PM10 effect in percentage points;
**negative = larger PM10 discount** in that group.

| Specification | PM10 × near a main road | PM10 × noise 65 dB+ |
|---|---|---|
{chr(10).join(irows)}

\\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01. Both interactions vary within an LSOA-year, so they can be
estimated even with LSOA-by-year fixed effects, which remove any local shock.

- **PM10 × near:** {mtxt(i_near, 'pm10_x_near')}.
- **PM10 × loud:** {mtxt(i_loud, 'pm10_x_loud')}.

## Part 3: Does the flat discount come from flats on main roads?

`model_v2.md` found a much larger PM10 discount for flats than for houses. If that is because flats are more often
on main roads, the house interaction should shrink once road proximity and noise are controlled for.

| Model | PM10 × house | Std. error |
|---|---|---|
| Model v2 (same sales) | {c_h0 * 100:+.3f}{stars(c_h0, s_h0)} | {s_h0 * 100:.3f} |
| + road and noise bands | {c_h1 * 100:+.3f}{stars(c_h1, s_h1)} | {s_h1 * 100:.3f} |
| + road and noise bands + PM10 × near + PM10 × loud | {c_h2 * 100:+.3f}{stars(c_h2, s_h2)} | {s_h2 * 100:.3f} |

**PM10 effect for four groups** (one model, each group has its own PM10 coefficient):

| Group | PM10 effect | Std. error (pp) | WTP per µg/m³ | Sales |
|---|---|---|---|---|
{chr(10).join(grp_rows)}

Difference-in-differences, (flats near − flats far) − (houses near − houses far): {b_dd * 100:+.3f} pp
(s.e. {se_dd * 100:.3f}, p {p_fmt(p_normal(b_dd / se_dd))}).

## Verdict

**1. Main roads and noise lower prices, separately from each other.** Both blocks are jointly significant even
when the other is in the model (roads given noise p {p_fmt(p_r)}, noise given roads p {p_fmt(p_n)}).

- Homes within 50 m of a main road sell for {price_pct(coef(m_road, 'road_0_50')[0])} compared with homes 400 m
  or more away ({price_pct(coef(m_both, 'road_0_50')[0])} given noise). Beyond 50 m there is no clear gradient,
  which fits the fast decline of traffic pollution and noise with distance. Next to the strategic network
  (motorways and primary routes) the discount is larger: {price_pct(coef(m_primary, 'road_0_50')[0])}.
- Noise of 65-69.9 dB goes with {price_pct(coef(m_both, 'noise_65')[0])} given road distance. The {noise_n.get(75, 0):,}
  sales at 75 dB or more show {price_pct(coef(m_both, 'noise_75')[0])}, but they come from very few streets, so we
  do not read this as a precise estimate. Below 65 dB noise does not matter.
- With LSOA-by-year fixed effects, which compare homes sold in the same LSOA in the same year, the estimates are
  of similar size: 0-50 m {mtxt(m_ly, 'road_0_50').replace(' pp', '')}, 65-69.9 dB
  {mtxt(m_ly, 'noise_65').replace(' pp', '')} (×100).

**2. The PM10 effect does not change** ({pct(b_v2)} → {pct(b_both)}). Roads and noise are fixed over time, while
PM10 with LSOA fixed effects is identified from changes over time, so the two do not compete. They measure
different things: road proximity is the *local* exposure that the monitoring sites cannot see, PM10 the
area-wide level and its changes.

**3. No larger PM10 discount next to main roads or in noisy places.** PM10 × near is
{mtxt(i_near, 'pm10_x_near')} and PM10 × loud {mtxt(i_loud, 'pm10_x_loud')}, and neither becomes significant in any
check, including LSOA-by-year fixed effects.

**4. The flat discount is not about main roads.** Flats are much more often near a main road
({by_type.at[0, 'near'] * 100:.0f}% vs {by_type.at[1, 'near'] * 100:.0f}% of houses), but the PM10 effect is the
same for flats near and far from one ({pct(coef(g, 'pm10_flat_near')[0])} and {pct(coef(g, 'pm10_flat_far')[0])}),
and the same for houses near and far ({pct(coef(g, 'pm10_house_near')[0])} and {pct(coef(g, 'pm10_house_far')[0])}).
The house interaction does not shrink with the road and noise controls ({c_h0 * 100:+.3f} → {c_h2 * 100:+.3f} pp).
So explanation (1) in `model_v2.md` is not supported at postcode precision. What is left: flats have fewer ways
to avoid the outside air (one facade, no garden, ground-floor flats at street level), or flat buyers differ.
Floor level (EPC data) is the next test that can separate these.

## Limitations

- **Postcode centroids, not addresses.** Distances are measured from the centre of the postcode, so a home on the
  main road and one around the corner can get the same value. This blurs the 0-50 m band most.
- **Roads today, noise in 2012.** The road network is the current one and the noise map is from 2012, while the
  sales are from 2007-2014. Major London roads changed little in that time.
- **Bundled effects.** Living next to a main road also means traffic, bus routes, shops and less privacy. Noise
  separates one part of this; the rest stays in the distance bands.
- **Not causal.** Like model v2, these are associations conditional on the controls and fixed effects.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE}")


if __name__ == "__main__":
    main()
