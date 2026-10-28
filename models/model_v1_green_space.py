"""
HE3027 Group Project: does green space change how much PM10 lowers house prices?

Two opposite ideas, so the test is two-sided:
  - Buffer: trees and grass catch particles and a park nearby offers cleaner air, so homes in green
    surroundings should lose LESS value when PM10 rises (positive PM10 × green interaction).
  - Exposure: people who live next to green space spend more time outdoors, where PM10 is breathed in,
    so they should care MORE about it (negative interaction).

Green space comes from OS Open Greenspace (public parks, playing fields, cemeteries, allotments, ...). For
every postcode we measure
  1. the share of the land within 500 m that is green space (main measure; 250 m and 1 km as checks),
  2. whether a local park of 2 ha or more is within 400 m (the London Plan standard for access to open space),
and split the sales into terciles of green share (low / middle / high green).

Builds on the v1 model (M8: LSOA and year-quarter fixed effects, MSOA-clustered standard errors) and adds
PM10 × green space, with checks for the main confounders (greener postcodes have more houses and lie further out).

Writes model_v1_green_space.md and figures/green_space.png. The green-space values per postcode are cached in
data/green_features.csv, which is built on the first run (delete it to rebuild).

Run from the repository root:
    python models/model_v1_green_space.py

Needs: pandas, numpy, matplotlib, scipy, shapely, pyproj, pyshp
Data:  data/land_registry_pollution.dta
       data/os_open_greenspace/opgrsp_essh_tq.zip and opgrsp_essh_tl.zip   (keep zipped)
           OS Open Greenspace, ESRI Shapefile, 100 km squares TQ and TL:
           https://osdatahub.os.uk/downloads/open/OpenGreenspace
"""

import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
import shapely
from pyproj import Transformer
from scipy.signal import fftconvolve

from model_v1 import (ACCESS, AMENITY, CLUSTER, CODE_DIR, MODEL_DIR, DATA_DIR, PROPERTY, SOCIO, fe_ols, load_data, pct,
                      stars, two_fe_ols)
from model_v1_income_groups import gbp, p_fmt, p_normal, wald
from model_v1_income_small_area import lincom
from model_v1_pm10_smoothing import add_smoothed_pm10
from model_v2 import by_year, coef, sig_txt, varying

MD_FILE = MODEL_DIR / "model_v1_green_space.md"
FIG_DIR = CODE_DIR / "figures"
FIG_FILE = FIG_DIR / "green_space.png"

GREEN_DIR = DATA_DIR / "os_open_greenspace"
GREEN_TILES = ["TQ", "TL"]       # 100 km grid squares: London lies in TQ, its northern edge reaches into TL
FEATURES = DATA_DIR / "green_features.csv"
ROAD_FEATURES = DATA_DIR / "road_features.csv"   # from model_v2_roads.py; used for one check if it exists

CONTROLS = PROPERTY + ACCESS + AMENITY + SOCIO     # the M8 controls

# OS Open Greenspace `function`. Tennis courts and "other sports facilities" (leisure centres, stadiums) are
# mostly hard surfaces or buildings, so they do not count as green.
GREEN_TYPES = ["Public Park Or Garden", "Playing Field", "Golf Course", "Cemetery",
               "Allotments Or Community Growing Spaces", "Religious Grounds", "Play Space", "Bowling Green"]
PARK_TYPES = ["Public Park Or Garden"]
LOCAL_PARK_HA = 2        # London Plan: local parks of 2 ha or more ...
LOCAL_PARK_M = 400       # ... within 400 m of home
RADII = [250, 500, 1000]  # m; 500 m is the main measure
RES_M = 20               # raster cell size used to measure green shares (m)
BOX = (495_000, 570_000, 150_000, 210_000)   # British National Grid (easting, northing) around London

GROUPS = {"low": "Low green", "mid": "Middle green", "high": "High green"}   # terciles; base group: low
COLORS = {"low": "#8a8984", "mid": "#7fc7a5", "high": "#1baf7a"}


# ---------------------------------------------------------------------------
# Data: green space per postcode
# ---------------------------------------------------------------------------
def read_greenspace():
    """All green-space sites in the London box as shapely polygons (British National Grid) and their type."""
    polys, kinds = [], []
    for tile in GREEN_TILES:
        with zipfile.ZipFile(GREEN_DIR / f"opgrsp_essh_{tile.lower()}.zip") as z:
            base = next(n[:-4] for n in z.namelist() if n.endswith(f"{tile}_GreenspaceSite.shp"))
            r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), shx=io.BytesIO(z.read(base + ".shx")),
                                 dbf=io.BytesIO(z.read(base + ".dbf")), encoding="latin-1")
            for shp, rec in zip(r.iterShapes(), r.iterRecords(fields=["function"])):
                polys.append(shapely.geometry.shape(shp.__geo_interface__))
                kinds.append(rec[0])
    polys, kinds = np.array(polys), np.array(kinds)
    e0, e1, n0, n1 = BOX
    keep = shapely.intersects(polys, shapely.box(e0, n0, e1, n1))
    return polys[keep], kinds[keep]


def rasterize(polys):
    """Boolean grid (RES_M cells over BOX): True where the cell centre lies in one of the polygons.
    Overlapping sites (a play space inside a park) are counted once."""
    e0, e1, n0, n1 = BOX
    nx, ny = int((e1 - e0) / RES_M), int((n1 - n0) / RES_M)
    grid = np.zeros((ny, nx), dtype=bool)
    for p in polys:
        a, b, c, d = p.bounds
        i0, i1 = max(int((a - e0) / RES_M), 0), min(int((c - e0) / RES_M) + 1, nx)
        j0, j1 = max(int((b - n0) / RES_M), 0), min(int((d - n0) / RES_M) + 1, ny)
        if i0 >= i1 or j0 >= j1:
            continue
        X, Y = np.meshgrid(e0 + (np.arange(i0, i1) + 0.5) * RES_M, n0 + (np.arange(j0, j1) + 0.5) * RES_M)
        shapely.prepare(p)
        grid[j0:j1, i0:i1] |= shapely.contains_xy(p, X, Y)
    return grid


def share_within(grid, e, n, radius):
    """Share of the area within `radius` m of each point that is green (moving disc average of the grid)."""
    k = int(round(radius / RES_M))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disc = (xx ** 2 + yy ** 2 <= k ** 2).astype(float)
    smooth = fftconvolve(grid.astype(float), disc / disc.sum(), mode="same")
    e0, _, n0, _ = BOX
    i = np.clip(((e - e0) / RES_M).astype(int), 0, grid.shape[1] - 1)
    j = np.clip(((n - n0) / RES_M).astype(int), 0, grid.shape[0] - 1)
    return np.clip(smooth[j, i], 0, 1)


def build_features(df):
    pc = df.groupby("postcode", as_index=False)[["lat", "lon"]].first()
    e, n = Transformer.from_crs(4326, 27700, always_xy=True).transform(pc["lon"].to_numpy(), pc["lat"].to_numpy())
    polys, kinds = read_greenspace()
    green = np.isin(kinds, GREEN_TYPES)
    park = np.isin(kinds, PARK_TYPES)
    print(f"  {green.sum():,} green-space sites, {park.sum():,} of them public parks or gardens")

    g_all, g_park = rasterize(polys[green]), rasterize(polys[park])
    for r in RADII:
        pc[f"green_{r}"] = share_within(g_all, e, n, r)
    pc["park_500"] = share_within(g_park, e, n, 500)

    local = polys[park & (shapely.area(polys) >= LOCAL_PARK_HA * 10_000)]
    _, dist = shapely.STRtree(local).query_nearest(shapely.points(e, n), return_distance=True, all_matches=False)
    pc["dist_local_park_m"] = dist          # 0 if the postcode lies inside the park
    pc.drop(columns=["lat", "lon"]).to_csv(FEATURES, index=False)
    return pc


def add_green(df):
    if FEATURES.exists():
        pc = pd.read_csv(FEATURES)
    else:
        print("Building data/green_features.csv ...")
        pc = build_features(df)
    # round, so the values are identical whether they come from the first run or from the cached CSV
    # (many postcodes share a share exactly at a tercile cut, so the last digits would decide their group)
    shares = [f"green_{r}" for r in RADII] + ["park_500"]
    pc[shares] = pc[shares].round(6)
    df = df.merge(pc, on="postcode", how="left")
    if ROAD_FEATURES.exists():   # main roads and road noise, built by model_v2_roads.py
        rd = pd.read_csv(ROAD_FEATURES, usecols=["postcode", "dist_major_m", "noise_lden"])
        df = df.merge(rd, on="postcode", how="left")
        df["near_road"] = (df["dist_major_m"] < 100).astype(int)
        df["loud"] = (df["noise_lden"] >= 65).astype(int)
    df["near_park"] = (df["dist_local_park_m"] <= LOCAL_PARK_M).astype(int)
    df["house"] = df[["detached_dum", "semi_d_dum", "terrace_dum"]].sum(axis=1)
    df["ln_cbd"] = np.log(df["dist_to_cbd"].clip(lower=0.01))
    df["lsoa_year"] = df["lsoacode"] + "_" + df["year"].astype(str)
    df["la_year"] = df["oslaua"] + "_" + df["year"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Interaction terms
# ---------------------------------------------------------------------------
def with_pm(d, pm="pm10", green="green_500"):
    """Sample with `pm` as the PM10 measure and `green` as the green-space measure, plus all interaction terms.

    All moderators are centred at the sample mean, so the `pm10` coefficient is the PM10 effect at average green
    space and the average mix of houses and new builds. `green10` is the green share in units of 10 percentage
    points. The terciles are cut over the sales of this sample (each holds about a third of the sales).
    """
    d = d.dropna(subset=[pm, green]).copy()
    d["pm10"] = d[pm]
    d["green10"] = (d[green] - d[green].mean()) / 0.10
    d["pm10_x_green"] = d["pm10"] * d["green10"]
    d["pm10_x_near"] = d["pm10"] * (d["near_park"] - d["near_park"].mean())
    q1, q2 = d[green].quantile([1 / 3, 2 / 3])
    d["green_grp"] = np.where(d[green] < q1, "low", np.where(d[green] < q2, "mid", "high"))
    for k in ["mid", "high"]:
        d[f"grp_{k}"] = (d["green_grp"] == k).astype(int)
        d[f"pm10_x_{k}"] = d["pm10"] * d[f"grp_{k}"]
    d["pm10_x_house"] = d["pm10"] * (d["house"] - d["house"].mean())
    d["pm10_x_new"] = d["pm10"] * (d["newbuild"] - d["newbuild"].mean())
    d["pm10_x_cbd"] = d["pm10"] * (d["ln_cbd"] - d["ln_cbd"].mean())
    for v in ["near_road", "loud"]:
        if v in d:
            d[f"pm10_x_{v}"] = d["pm10"] * (d[v] - d[v].mean())
    return d


CONT = ["pm10_x_green", "green10"]                          # green share within 500 m
NEAR = ["pm10_x_near", "near_park"]                         # local park within 400 m
GRP = ["pm10_x_mid", "pm10_x_high", "grp_mid", "grp_high"]  # terciles (base: low green)
PTYPE = ["pm10_x_house", "pm10_x_new"]                      # property-type confounders
SPECS = {"cont": CONT, "near": NEAR, "grp": GRP}


def m8(d, extra, cluster=CLUSTER, ptype=True):
    """M8 + green-space terms (+ PM10 × house and PM10 × new build). Terms that do not vary are dropped."""
    pt = [t for t in PTYPE if ptype and not (t == "pm10_x_house" and d["house"].nunique() == 1)]
    xs = [v for v in ["pm10"] + list(extra) + pt + CONTROLS if d[v].std() > 0]
    return fe_ols(d, xs, time_fe=True, area_fe="lsoacode", cluster=cluster)


def m8_trends(d, extra):
    """Green-space-specific year effects: the green measure of the model gets its own year dummies
    (terciles: their own year-quarter dummies), so a different price trend in green areas cannot show up
    as an interaction."""
    d = d.copy()
    if "grp_high" in extra:
        d["yearquarter"] = d["green_grp"] + "_" + d["yearquarter"].astype(str)
        return m8(d, extra)
    col = "green10" if "green10" in extra else "near_park"
    return m8(d, list(extra) + by_year(d, col))


def m8_la_year(d, extra):
    return two_fe_ols(d, ["pm10"] + list(extra) + PTYPE + CONTROLS, "lsoacode", "la_year")


def m8_lsoa_year(d, extra):
    """LSOA-by-year fixed effects: compares green and less green postcodes in the same LSOA and year."""
    return fe_ols(d, varying(d, ["pm10"] + list(extra) + PTYPE + CONTROLS, "lsoa_year"), time_fe=True,
                  area_fe="lsoa_year")


def fit_all(fn, d, which=("cont", "near", "grp")):
    return {k: fn(d, SPECS[k]) for k in which}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TERM = {"cont": "pm10_x_green", "near": "pm10_x_near", "grp": "pm10_x_high"}


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
    """PM10 effect, its s.e., WTP and mean price for each green tercile from a tercile model."""
    out = {}
    for k in GROUPS:
        w = {"pm10": 1} if k == "low" else {"pm10": 1, f"pm10_x_{k}": 1}
        if not all(has(m, t) for t in w):
            continue
        c, s = lincom(m, w)
        p = d.loc[d["green_grp"] == k, "price"].mean()
        out[k] = {"coef": c, "se": s, "price": p, "wtp": -c * p, "n": int((d["green_grp"] == k).sum())}
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
    titles = [t for t, _ in panels]
    rows = [(title, k, e[k]) for title, e in panels for k in GROUPS if k in e]
    y = np.arange(len(rows))[::-1].astype(float) - 0.6 * np.array([titles.index(r[0]) for r in rows])
    ax.axvline(0, color="#52514e", linewidth=0.8)
    for yi, (title, k, e) in zip(y, rows):
        ax.errorbar(e["coef"] * 100, yi, xerr=1.96 * e["se"] * 100, fmt="o", color=COLORS[k], ecolor="#0b0b0b",
                    elinewidth=1, capsize=3, markersize=7, zorder=3)
        ax.annotate(f"{e['coef'] * 100:+.2f}%", (e["coef"] * 100, yi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color="#0b0b0b")
    ax.set_yticks(y, [f"{title}: {GROUPS[k]}" for title, k, _ in rows])
    ax.set_xlabel("% change in price per +1 µg/m³ PM10 (95% CI)")
    ax.set_title("PM10 effect by green space within 500 m of the postcode (v1 model)", loc="left", fontsize=11,
                 color="#0b0b0b")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Low / middle / high green: terciles of the green share within 500 m (OS Open Greenspace). "
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
    df = add_green(df)
    d = with_pm(df)
    print(f"Sales: {len(d):,}")

    # ===================== Part 1: the green premium (no interaction) =====================
    prem = {"cont": m8(d, ["green10"], ptype=False), "near": m8(d, ["near_park"], ptype=False),
            "grp": m8(d, ["grp_mid", "grp_high"], ptype=False)}

    # ===================== Part 2: does the PM10 effect depend on green space? =====================
    main_fits = fit_all(m8, d)
    c0, s0 = coef(main_fits["cont"], "pm10_x_green")
    print(f"main: PM10 x green = {c0 * 100:+.3f} pp (se {s0 * 100:.3f})")

    flats, houses = d[d["house"] == 0], d[d["house"] == 1]
    d_ma3 = with_pm(df, pm="pm10_ma3")
    alt = {f"green_{r}": with_pm(df, green=f"green_{r}") for r in RADII if r != 500}
    d_park = with_pm(df, green="park_500")

    checks = [  # (label, {spec: model})
        ("**Main: M8 + green space + PM10 × house + PM10 × new build**", main_fits),
        ("Without PM10 × house and PM10 × new build", fit_all(lambda x, e: m8(x, e, ptype=False), d)),
        ("+ PM10 × log distance to the centre", fit_all(lambda x, e: m8(x, list(e) + ["pm10_x_cbd"]), d)),
    ]
    if "near_road" in d:
        road_terms = ["pm10_x_near_road", "pm10_x_loud", "near_road", "loud"]
        checks.append(("+ PM10 × near a main road and PM10 × road noise 65 dB+",
                       fit_all(lambda x, e: m8(x, list(e) + road_terms), d)))
    checks += [
        ("+ green-specific year effects", fit_all(m8_trends, d)),
        ("+ borough-by-year fixed effects", fit_all(m8_la_year, d)),
        ("LSOA-by-year fixed effects", fit_all(m8_lsoa_year, d)),
        ("Standard errors clustered by borough", fit_all(lambda x, e: m8(x, e, cluster="oslaua"), d)),
        ("3-year average PM10 (v2 measure)", fit_all(m8, d_ma3)),
        ("Green share within 250 m instead of 500 m", fit_all(m8, alt["green_250"], ("cont", "grp"))),
        ("Green share within 1 km instead of 500 m", fit_all(m8, alt["green_1000"], ("cont", "grp"))),
        ("Only public parks and gardens (share within 500 m)", fit_all(m8, d_park, ("cont", "grp"))),
        ("Flats only", fit_all(m8, flats)),
        ("Houses only", fit_all(m8, houses)),
    ]
    print("robustness done")

    # --- PM10 effect and WTP by tercile ---
    eff_all = group_effects(main_fits["grp"], d)
    eff_flat = group_effects(checks[-2][1]["grp"], flats)
    eff_house = group_effects(checks[-1][1]["grp"], houses)
    plot([("All homes", eff_all), ("Flats", eff_flat), ("Houses", eff_house)])

    # --- tests and summary numbers ---
    mc, mn, mg = main_fits["cont"], main_fits["near"], main_fits["grp"]
    W_eq, _, p_eq = wald(mg, ["pm10_x_mid", "pm10_x_high"])
    b_int, se_int = coef(mc, "pm10_x_green")
    b_pm, se_pm = coef(mc, "pm10")
    lo95, hi95 = b_int - 1.96 * se_int, b_int + 1.96 * se_int
    g_mu = d["green_500"].mean()
    at_q = {q: (d["green_500"].quantile(q),
                lincom(mc, {"pm10": 1, "pm10_x_green": (d["green_500"].quantile(q) - g_mu) / 0.10}))
            for q in [0.1, 0.5, 0.9]}

    def tally(spec):
        ts = [coef(f[spec], TERM[spec])[0] / coef(f[spec], TERM[spec])[1] for _, f in checks
              if spec in f and has(f[spec], TERM[spec])]
        return len(ts), sum(t > 1.96 for t in ts), sum(t < -1.96 for t in ts)

    tallies = {k: tally(k) for k in SPECS}
    sig_specs = [k for k in SPECS if abs(coef(main_fits[k], TERM[k])[0] / coef(main_fits[k], TERM[k])[1]) > 1.96]

    # ===================== Markdown =====================
    terc = d.groupby("green_grp")["green_500"].agg(["min", "max"])
    grp_desc = []
    for k, label in GROUPS.items():
        g = d[d["green_grp"] == k]
        grp_desc.append(f"| {label} | {terc.at[k, 'min'] * 100:.0f}% to {terc.at[k, 'max'] * 100:.0f}% | {len(g):,} | "
                        f"{g['green_500'].mean() * 100:.1f}% | {g['near_park'].mean() * 100:.0f}% | "
                        f"{(1 - g['house']).mean() * 100:.0f}% | {g['pm10'].mean():.1f} | "
                        f"{g['dist_to_cbd'].mean():.1f} | £{g['price'].mean():,.0f} |")
    corr = d[["green_500", "pm10", "house", "dist_to_cbd", "park_km"]].corr()["green_500"]
    flat_lo, flat_hi = (1 - d.loc[d["green_grp"] == k, "house"].mean() for k in ["low", "high"])
    if "near_road" in d:
        r_lo, r_hi = (d.loc[d["green_grp"] == k, "near_road"].mean() for k in ["low", "high"])
        road_txt = f": {r_lo * 100:.0f}% within 100 m of a main road vs {r_hi * 100:.0f}% in the greenest third"
    else:
        road_txt = "; this check needs data/road_features.csv from model_v2_roads.py and is skipped without it"
    within_sd = (d["green_500"] - d.groupby("lsoacode")["green_500"].transform("mean")).std()

    rob_md = []
    for label, f in checks:
        n = next(iter(f.values()))["n"]
        rob_md.append(f"| {label} | {cell(f.get('cont'), 'pm10_x_green')} | {cell(f.get('near'), 'pm10_x_near')} | "
                      f"{cell(f.get('grp'), 'pm10_x_high')} | {n:,} |")

    def eff_rows(title, e):
        return [f"| {title} | {GROUPS[k]} | {pct(x['coef'])}{stars(x['coef'], x['se'])} | {x['se'] * 100:.3f} | "
                f"{gbp(x['wtp'])} | £{x['price']:,.0f} | {x['n']:,} |" for k, x in e.items()]

    eff_md = eff_rows("All homes", eff_all) + eff_rows("Flats", eff_flat) + eff_rows("Houses", eff_house)
    q_md = [f"| P{int(q * 100)} | {g * 100:.1f}% | {pct(c)}{stars(c, s)} | {s * 100:.3f} |" for q, (g, (c, s)) in
            at_q.items()]

    pc_c, ps_c = coef(prem["cont"], "green10")
    pc_n, ps_n = coef(prem["near"], "near_park")
    pc_h, ps_h = coef(prem["grp"], "grp_high")

    noptype_c = coef(checks[1][1]["cont"], "pm10_x_green")
    cbd_c = coef(checks[2][1]["cont"], "pm10_x_green")
    flat_c = coef(checks[-2][1]["cont"], "pm10_x_green")
    house_c = coef(checks[-1][1]["cont"], "pm10_x_green")

    names = {"cont": "green share within 500 m (main measure)", "near": "local park within 400 m",
             "grp": "high vs low green tercile"}
    p_main = {k: p_normal(coef(main_fits[k], TERM[k])[0] / coef(main_fits[k], TERM[k])[1]) for k in SPECS}
    bonf = 0.05 / len(SPECS)
    tally_md = []
    for k in SPECS:
        n_k, pos_k, neg_k = tallies[k]
        c_k = coef(main_fits[k], TERM[k])[0]
        tally_md.append(f"| {names[k]} | {c_k * 100:+.3f} pp | {p_fmt(p_main[k])} | "
                        f"{'yes' if p_main[k] < bonf else 'no'} | {pos_k} of {n_k} | {neg_k} of {n_k} |")
    signs = {np.sign(coef(main_fits[k], TERM[k])[0]) for k in sig_specs}
    robust = [k for k in sig_specs if max(tallies[k][1], tallies[k][2]) >= 2 / 3 * tallies[k][0]]
    lsoa_year_grp = coef(dict(checks)["LSOA-by-year fixed effects"]["grp"], "pm10_x_high")
    main_grp = coef(mg, "pm10_x_high")
    lsoa_year_txt = (f"With LSOA-by-year fixed effects the tercile contrast is {lsoa_year_grp[0] * 100:+.3f} pp "
                     f"({sig_txt(*lsoa_year_grp)}, standard error {lsoa_year_grp[1] * 100:.3f} vs "
                     f"{main_grp[1] * 100:.3f} in the main model).")
    idea = "buffer" if signs == {1.0} else "exposure" if signs == {-1.0} else None
    if not sig_specs:
        verdict = ("**No effect.** None of the three green-space measures changes the PM10 effect significantly "
                   "(two-sided, 5%). Neither the buffer idea nor the exposure idea is supported.")
    elif "cont" in sig_specs and "cont" in robust and idea:
        verdict = (f"**{idea.capitalize()} idea supported.** The main measure is significant and holds in most "
                   f"checks: homes with more green space around them lose "
                   f"{'less' if idea == 'buffer' else 'more'} value when PM10 rises.")
    elif idea:
        verdict = (f"**Weak evidence for the {idea} idea.** The main measure (green share within 500 m) points the same "
                   f"way but is {'significant' if 'cont' in sig_specs else 'not significant'} at 5% "
                   f"(p {p_fmt(p_main['cont'])}). The clearer signal is in the "
                   + " and ".join(f"{names[k]} (significant at 5% in {max(tallies[k][1], tallies[k][2])} of "
                                  f"{tallies[k][0]} specifications)" for k in sig_specs if k != "cont") +
                   f". The local-park measure shows {'nothing' if 'near' not in sig_specs else 'the same'}.")
    else:
        verdict = "**Mixed.** The green-space measures point in different directions; no clear conclusion."
    q_low, q_high = (d.loc[d["green_grp"] == k, "green_500"] for k in ["low", "high"])
    max_change = max(abs(lo95), abs(hi95)) * (d["green_500"].quantile(0.9) - d["green_500"].quantile(0.1)) / 0.10
    size_txt = (f"Going from the 10th to the 90th percentile of green share changes the PM10 effect by at most "
                f"{max_change * 100:.2f} pp (end of the 95% interval furthest from zero), compared with a PM10 "
                f"effect of {pct(b_pm)}.")

    md = f"""# Model v1 + Green Space: Does Green Space Change the PM10 Effect?

Generated by `model_v1_green_space.py`. All numbers in this file are produced by the script.
Outcome: log sale price. Coefficients ×100 give the % change in price.

## Question and hypotheses

Does the price discount for PM10 depend on how much green space there is around a home? There are two opposite
ideas, so we test **two-sided**:

| Idea | Mechanism | Prediction for PM10 × green |
|---|---|---|
| **Buffer** | Trees and grass catch particles, and a park nearby is a place with cleaner air | Positive: a smaller discount in green surroundings |
| **Exposure** | People who live next to green space spend more time outdoors, where they breathe the PM10 | Negative: a larger discount in green surroundings |

We fix the test before looking at the results: the main specification below, two-sided at 5%, and the robustness
checks in the table.

## Data

| Data | What we use | Level | Year | Source |
|---|---|---|---|---|
| House sales, PM10, controls | As in the v1 model (M8) | Each sale | 2005-2014 | Course data set `land_registry_pollution.dta` |
| **Green space (new)** | Outlines of public parks and gardens, playing fields, golf courses, cemeteries, allotments, religious grounds, play spaces and bowling greens | Each site (polygon) | Current release (London's parks are protected and changed little since 2005-2014) | [OS Open Greenspace](https://osdatahub.os.uk/downloads/open/OpenGreenspace), 100 km squares TQ and TL, Open Government Licence |

Tennis courts and other sports facilities (leisure centres, stadiums) are left out because they are mostly hard
surfaces or buildings. Overlapping sites (a play space inside a park) are counted once.

**Why not the park variable in the course data?** `Nearest_Park_dist` is the distance to the nearest *Grade I*
registered park, a few large historic parks. It is already one of the M8 controls, but it mostly measures how
central a home is (correlation with the distance to the centre in our sample: {d[['park_km', 'dist_to_cbd']].corr().iloc[0, 1]:.2f}),
not how much green space is around it (correlation with the green share: {corr['park_km']:.2f}).

### Green-space measures per postcode

| Measure | Definition | Mean | Use |
|---|---|---|---|
| Green share within 500 m | Share of the land within 500 m of the postcode centroid that is green space | {d['green_500'].mean() * 100:.1f}% | Main measure, per +10 pp |
| Local park within 400 m | A public park or garden of {LOCAL_PARK_HA} ha or more within {LOCAL_PARK_M} m (London Plan standard for access to a local park) | {d['near_park'].mean() * 100:.0f}% of sales | Second measure (0/1) |
| Terciles | Low / middle / high green: thirds of the sales by green share within 500 m | | Groups, PM10 effect per group |
| Green share within 250 m and 1 km, public parks only | Same as the main measure with another radius or only parks and gardens | | Checks |

Shares are measured on a {RES_M} m grid. The green share varies a lot within LSOAs (standard deviation
{within_sd * 100:.1f} pp within LSOAs, {d['green_500'].std() * 100:.1f} pp overall), which is what the LSOA
fixed-effects model needs. All {len(d):,} sales of the v1 sample get the measures.

| Group | Green share within 500 m | Sales | Mean green share | Local park within 400 m | Flats | PM10 (µg/m³) | Distance to centre (km) | Mean price |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(grp_desc)}

Correlation of the green share with: PM10 {corr['pm10']:.2f}, house {corr['house']:.2f}, distance to the centre
{corr['dist_to_cbd']:.2f}.

**Possible confounders.** The greenest third has fewer flats ({flat_hi * 100:.0f}% vs {flat_lo * 100:.0f}% in the
least green third). Houses lose less value from PM10 than flats (`model_v2.md`), so a simple PM10 × green term could
partly pick up PM10 × house. The main specification therefore includes **PM10 × house** and **PM10 × new build**;
the checks add **PM10 × distance to the centre**, **PM10 × main road and road noise** (dense places with little green
space may lie on busy roads{road_txt}) and estimate flats and houses separately.

## Part 1: Do homes with more green space sell for more?

Within an LSOA and quarter (M8 controls, no PM10 interaction):

| Term | Effect on price | Std. error |
|---|---|---|
| +10 pp green share within 500 m | {pct(pc_c)}{stars(pc_c, ps_c)} | {ps_c * 100:.2f} |
| Local park ({LOCAL_PARK_HA} ha+) within {LOCAL_PARK_M} m | {pct(pc_n)}{stars(pc_n, ps_n)} | {ps_n * 100:.2f} |
| High vs low green tercile | {pct(pc_h)}{stars(pc_h, ps_h)} | {ps_h * 100:.2f} |

Hedonic studies usually find a premium for living near green space; this is a by-product here. Green space does not
change over time, so with LSOA fixed effects it is identified only from differences between postcodes in the same
LSOA. Most of the premium is between neighbourhoods and is absorbed by the fixed effects, so these within-LSOA
estimates are small and should not be read as the full value of green space.

## Part 2: Does the PM10 effect depend on green space?

ln(price) = β · PM10 + γ · PM10 × green + δ · green + θ₁ · PM10 × house + θ₂ · PM10 × new build
+ M8 controls + LSOA fixed effects + year-quarter fixed effects + error

Green space, house and new build are centred, so β is the PM10 effect at average green space and the average mix of
property types. **γ > 0: smaller discount in green surroundings (buffer); γ < 0: larger discount (exposure).**
PM10 changes over time within an LSOA, so γ asks whether prices in green postcodes react differently to the
*same change* in PM10 than prices in less green postcodes of the same LSOA.

### Main result

- **PM10 effect at average green space:** {pct(b_pm)}{stars(b_pm, se_pm)} per µg/m³ (standard error {se_pm * 100:.3f} pp).
- **PM10 × green share:** {mtxt(mc, 'pm10_x_green')} per +10 pp green share within 500 m.
  95% confidence interval [{lo95 * 100:+.3f}, {hi95 * 100:+.3f}] pp.
- **PM10 × local park within 400 m:** {mtxt(mn, 'pm10_x_near')} (with a park minus without).
- **Terciles:** high minus low green {mtxt(mg, 'pm10_x_high')}; middle minus low green {mtxt(mg, 'pm10_x_mid')}.
  Joint test that the PM10 effect is the same in all three terciles: W = {W_eq:.2f}, p = {p_fmt(p_eq)}.

PM10 effect at different levels of green share (main model):

| Green share within 500 m | Value | PM10 effect | Std. error (pp) |
|---|---|---|---|
{chr(10).join(q_md)}

### Robustness

| Specification | PM10 × green share (per +10 pp) | PM10 × local park within 400 m | PM10 × high − low green tercile | Sales |
|---|---|---|---|---|
{chr(10).join(rob_md)}

Coefficient ×100, in percentage points of the PM10 effect (standard error ×100). \\* p < 0.10, \\*\\* p < 0.05,
\\*\\*\\* p < 0.01. Empty cell: not estimated (the alternative green measures replace the green share and its terciles;
the local-park measure stays the same).

- **Without PM10 × house and PM10 × new build:** {noptype_c[0] * 100:+.3f} pp instead of {b_int * 100:+.3f} pp.
- **+ PM10 × log distance to the centre:** {cbd_c[0] * 100:+.3f} pp ({sig_txt(*cbd_c)}). Green postcodes lie further
  out, so this checks that the interaction is not just a centre-periphery difference in the PM10 effect.
- **Green-specific year effects:** green and less green postcodes get their own year effects, so a different price
  trend in green areas cannot show up as an interaction.
- **Borough-by-year fixed effects:** remove any shock that hits a whole borough in a given year.
- **LSOA-by-year fixed effects:** compare green and less green postcodes in the same LSOA in the same year, so any
  local shock cancels out. PM10 itself is absorbed; the interaction is still identified.
- **Flats only / houses only:** flats {flat_c[0] * 100:+.3f} pp ({sig_txt(*flat_c)}), houses
  {house_c[0] * 100:+.3f} pp ({sig_txt(*house_c)}).

### PM10 effect for each green tercile

| Homes | Green tercile | PM10 effect | Std. error (pp) | WTP per µg/m³ | Mean price | Sales |
|---|---|---|---|---|---|---|
{chr(10).join(eff_md)}

WTP = −effect × mean price of the group. "All homes" is the main tercile model; flats and houses are separate
models (with their own terciles cut over all sales).

![PM10 effect by green space](../figures/green_space.png)

## Verdict

{verdict}

| Measure | Main estimate | p (two-sided) | Passes Bonferroni for 3 measures (p < {bonf:.4f}) | Significantly positive (5%) in | Significantly negative (5%) in |
|---|---|---|---|---|---|
{chr(10).join(tally_md)}

- **Where the difference is.** The PM10 discount is {pct(eff_all['low']['coef'])} in the least green third of
  postcodes (green share below {q_low.max() * 100:.0f}% within 500 m) and {pct(eff_all['mid']['coef'])} /
  {pct(eff_all['high']['coef'])} in the middle and greenest thirds. So the pattern is less "the more green, the
  smaller the discount" than "almost no green space nearby goes with a larger discount", which is why the linear
  green-share term is weaker than the tercile contrast.
- **Size:** {size_txt}
- **Caution.** A tercile split and two other measures were tested; only the result that survives the checks in
  the table and the Bonferroni correction should be reported as more than a hint. {lsoa_year_txt}

## Limitations

- **Green space today, sales 2005-2014.** The green-space outlines are from the current release. London's parks are
  protected (Metropolitan Open Land), but a few large sites are new, for example the Queen Elizabeth Olympic Park
  (opened 2012-2014).
- **Public green space only.** Private gardens and street trees are not in OS Open Greenspace, and they are probably
  what matters most for catching particles right at the home.
- **Area, not exposure.** The green share says how green the surroundings are, not how much time residents spend
  outdoors or where the PM10 they breathe comes from.
- **PM10 from monitoring sites.** As in all our models, PM10 is the yearly mean at the nearest site. It cannot see
  whether the air inside a park is cleaner than on the street next to it, which is what the buffer idea is about.
  Interactions with a noisy PM10 are biased towards zero.
- **Sorting.** Households that value both green space and clean air may choose green, clean areas, so the
  interaction mixes preferences with who lives where.
- **Not causal.** Like M8, these are associations conditional on the controls and fixed effects.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE} and {FIG_FILE}")


if __name__ == "__main__":
    main()
