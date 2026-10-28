"""
HE3027 Group Project: model v3, the complete final model.

Collects everything we tried (model_v0 to model_v2_roads, plus the green-space and energy-efficiency analyses),
decides which controls the final model needs, and estimates it:

  1. Inventory: every control and every extra term we tested, with the decision for v3.
  2. Control audit: starting from a model with ALL candidate controls, each block is checked for
       - how much of its variation is left within LSOAs (the rest is absorbed by the LSOA fixed effects),
       - whether it is jointly significant,
       - how much the PM10 effect and the fit (BIC) change when the block is dropped.
  3. The v3 model: 3-year average PM10, the kept controls, main roads, road noise and green space (additional
     data), and two interactions, PM10 × house and PM10 × low green space, with robustness checks.

Decision rule for a block: keep it if it is jointly significant at 5% (MSOA-clustered Wald test) OR dropping it
moves the PM10 effect by 0.5 standard errors or more; then simplify a kept block to the terms that matter.

Writes model_v3.md and figures/v3_final.png.

Run from the repository root:
    python models/model_v3.py

Needs: pandas, numpy, matplotlib, scipy, shapely, pyproj, pyshp
Data:  data/land_registry_pollution.dta, data/gla_lsoa_atlas/lsoa-data.csv,
       data/road_features.csv      (built by model_v2_roads.py; built here if missing, needs the road data)
       data/green_features.csv     (built by model_v1_green_space.py; built here if missing, needs the green data)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model_v1_green_space as green
from model_v1 import CLUSTER, CODE_DIR, MODEL_DIR, PROPERTY, fe_ols, load_data, pct, stars, two_fe_ols
from model_v1_income_groups import bic, gbp, p_fmt, p_normal
from model_v1_income_small_area import lincom
from model_v1_pm10_smoothing import add_smoothed_pm10
from model_v2 import add_lsoa_atlas, add_v2_terms, coef, sig_txt, varying, with_pm
from model_v2_roads import NOISE_VARS, ROAD_VARS, add_road_features, price_pct, wald_chi2

MD_FILE = MODEL_DIR / "model_v3.md"
FIG_DIR = CODE_DIR / "figures"
FIG_FILE = FIG_DIR / "v3_final.png"

# ---------------------------------------------------------------------------
# Candidate controls, in blocks
# ---------------------------------------------------------------------------
BLOCKS = {  # key: (label, variables in the model with all candidates)
    "property": ("Property type, new build, tenure", PROPERTY),
    "far": ("Distance to centre, tube, Grade I park, Thames (log + log²)",
            [t for v in ["dist_to_cbd", "tube_km", "park_km", "thames_km"] for t in (f"ln_{v}", f"ln_{v}_sq")]),
    "bus": ("Bus stop distance (log + log²) and bus stops within 200 m",
            ["ln_bus_km", "ln_bus_km_sq", "Bus_stopscount_200m"]),
    "heritage": ("Grade I heritage buildings within 200 m", ["heritage_count_200m"]),
    "census": ("Census shares: no qualifications, minorities, lone parents",
               ["percent_noedu", "percent_minorities", "lone_hh"]),
    "borough": ("Borough economy by year: pay (cubic), job density, unemployment, population",
                ["pay_c", "pay_c2", "pay_c3", "jobdensity", "unemployment", "pop_k"]),
    "roads": ("Distance to a main road, 4 bands (OS Open Roads)", ROAD_VARS),
    "noise": ("Road noise, 5 bands (Defra 2012)", NOISE_VARS),
    "green": ("Green share within 500 m (OS Open Greenspace)", ["green10"]),
}
ALL_CONTROLS = [v for _, vs in BLOCKS.values() for v in vs]

KEEP_P = 0.05        # keep a block if jointly significant at this level ...
KEEP_SHIFT = 0.5     # ... or if dropping it moves the PM10 effect by this many standard errors

# The v3 controls: the blocks the rule keeps (see the audit in the markdown), with the bus, road and noise terms
# simplified to the parts that matter. Heritage and green space fail the rule.
V3_CONTROLS = (PROPERTY
               + BLOCKS["far"][1]
               + ["bus_km", "Bus_stopscount_200m"]
               + ["percent_noedu", "percent_minorities", "lone_hh"]
               + ["pay_c", "pay_c2", "pay_c3", "jobdensity", "unemployment", "pop_k"]
               + ["road_0_50", "noise_65_75", "noise_75"])
V3_BLOCKS = ["property", "far", "bus", "census", "borough", "roads", "noise"]
NEW_DATA = ["road_0_50", "noise_65_75", "noise_75"]
# The two interactions, and the green-space main effects that come with PM10 × low green (so that the interaction
# does not pick up level differences between green and less green postcodes)
INTERACTIONS = ["pm10_x_house", "pm10_x_lowgreen"]
GREEN_TERMS = ["lowgreen", "green10"]
MAIN = ["pm10"] + INTERACTIONS + V3_CONTROLS + GREEN_TERMS
# PM10 interactions tested across the project (see the inventory): house, PM10², age shares, education, income,
# near a main road, noise, green share, local park, green terciles, efficiency score, efficiency groups
N_INTERACTIONS = 12
LABELS = {
    "pm10": "PM10, 3-year average (µg/m³)",
    "pm10_x_house": "PM10 × house",
    "pm10_x_lowgreen": "PM10 × low green (least green third)",
    "lowgreen": "Low green: least green third of postcodes",
    "detached_dum": "Detached house (base: flat)",
    "semi_d_dum": "Semi-detached house",
    "terrace_dum": "Terraced house",
    "newbuild": "New build",
    "freehold": "Freehold (base: leasehold)",
    "bus_km": "Distance to nearest bus stop (km)",
    "Bus_stopscount_200m": "Bus stops within 200 m",
    "heritage_count_200m": "Grade I heritage buildings within 200 m",
    "percent_noedu": "Share without qualifications (0-1)",
    "percent_minorities": "Share ethnic minorities (0-1)",
    "lone_hh": "Share lone-parent households (0-1)",
    "pay_c": "Borough pay, centred (£1,000)",
    "pay_c2": "Borough pay²",
    "pay_c3": "Borough pay³",
    "jobdensity": "Borough job density",
    "unemployment": "Borough unemployment rate (%)",
    "pop_k": "Borough population (1,000)",
    "road_0_50": "Within 50 m of a motorway or A road",
    "noise_65_75": "Road noise 65-74.9 dB (base: below 65 dB)",
    "noise_75": "Road noise 75 dB or more",
    "green10": "Green share within 500 m (per +10 pp, centred)",
    "ln_dist_to_cbd": "log distance to centre", "ln_dist_to_cbd_sq": "log distance to centre²",
    "ln_tube_km": "log distance to tube", "ln_tube_km_sq": "log distance to tube²",
    "ln_park_km": "log distance to Grade I park", "ln_park_km_sq": "log distance to Grade I park²",
    "ln_thames_km": "log distance to Thames", "ln_thames_km_sq": "log distance to Thames²",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def add_green_share(df):
    """Green share within 500 m per postcode, from the cache of model_v1_green_space.py (built if missing)."""
    if green.FEATURES.exists():
        pc = pd.read_csv(green.FEATURES)
    else:
        print("Building data/green_features.csv ...")
        pc = green.build_features(df)
    pc["green_500"] = pc["green_500"].round(6)       # same rounding as model_v1_green_space.py
    return df.merge(pc[["postcode", "green_500"]], on="postcode", how="left")


def prepare():
    df, full, info = load_data()
    df, _ = add_smoothed_pm10(df, full)
    df, _ = add_lsoa_atlas(df)
    add_v2_terms(df)
    df = add_road_features(df)
    df = add_green_share(df)
    df["road_0_50"] = (df["dist_major_m"] < 50).astype(int)
    df["noise_65_75"] = df["noise_lden"].isin([65, 70]).astype(int)
    df["noise_75"] = (df["noise_lden"] >= 75).astype(int)
    df["green10"] = (df["green_500"] - df["green_500"].mean()) / 0.10
    return df


def finish(d):
    d["pm10_x_house"] = d["pm10"] * d["house"]
    low = d["green_500"] < d["green_500"].quantile(1 / 3)
    d["pm10_x_lowgreen"] = d["pm10"] * low
    d["lowgreen"] = low.astype(int)
    return d


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------
def fit(d, xs, area_fe="lsoacode", cluster=CLUSTER):
    return fe_ols(d, xs, time_fe=True, area_fe=area_fe, cluster=cluster)


def within_share(d, vs):
    """Share of the variance of the block's variables that is left after removing LSOA means (mean over vars)."""
    out = []
    for v in vs:
        x = d[v].astype(float)
        out.append((x - x.groupby(d["lsoacode"]).transform("mean")).var() / x.var())
    return float(np.mean(out))


def ctxt(m, t, scale=1.0):
    c, s = coef(m, t, scale)
    return f"{c * 100:+.3f} pp ({sig_txt(c, s)}, p {p_fmt(p_normal(c / s))})"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(rows):
    """rows: (label, coef, se, highlight)"""
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#52514e", linewidth=0.8)
    for yi, (lab, c, s, hi) in zip(y, rows):
        col = "#eb6834" if hi else "#2a78d6"
        ax.errorbar(c * 100, yi, xerr=1.96 * s * 100, fmt="o", color=col, ecolor="#0b0b0b", elinewidth=1,
                    capsize=3, markersize=7, zorder=3)
        ax.annotate(f"{c * 100:+.2f}%", (c * 100, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color="#0b0b0b")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("% change in price per +1 µg/m³ PM10 (95% CI)")
    ax.set_title("PM10 effect: model versions and the v3 model", loc="left", fontsize=11, color="#0b0b0b")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Orange: v3 main model. Blue: earlier versions and v3 checks. LSOA and year-quarter "
             "fixed effects in all models. Standard errors clustered by MSOA.", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = prepare()
    n_missing = int(df["green_500"].isna().sum() + df["dist_major_m"].isna().sum())
    d = finish(with_pm(df.dropna(subset=["green_500", "dist_major_m"]), "pm10_ma3"))      # v3 sample
    dy = finish(with_pm(df.dropna(subset=["green_500", "dist_major_m"]), "pm10"))         # yearly PM10
    mean_price = d["price"].mean()

    # ===================== Control audit =====================
    full = fit(d, ["pm10"] + ALL_CONTROLS)
    b_full, s_full = coef(full, "pm10")
    audit = []
    for key, (label, vs) in BLOCKS.items():
        m = fit(d, ["pm10"] + [v for v in ALL_CONTROLS if v not in vs])
        W, q, p = wald_chi2(full, vs)
        b = coef(m, "pm10")[0]
        shift = (b - b_full) / s_full
        audit.append((key, label, len(vs), within_share(d, vs), W, q, p, b, shift, bic(m) - bic(full),
                      p < KEEP_P or abs(shift) >= KEEP_SHIFT))
    kept = [a[0] for a in audit if a[-1]]
    if kept != V3_BLOCKS:
        print(f"WARNING: the rule keeps {kept}, but V3_CONTROLS is built from {V3_BLOCKS}")
    A = {a[0]: a for a in audit}
    print("audit done")
    # the bands that v3 leaves out: road 50-400 m and noise 55-64.9 dB
    W_rest, q_rest, p_rest = wald_chi2(full, ["road_50_100", "road_100_200", "road_200_400",
                                              "noise_55", "noise_60"])

    # ===================== v3 =====================
    def drop(xs, vs):
        return [x for x in xs if x not in vs]

    v3 = fit(d, MAIN)                                               # main model
    v3_avg = fit(d, ["pm10"] + V3_CONTROLS + GREEN_TERMS)           # average PM10 effect, no interactions
    v3_y = fit(dy, MAIN)                                            # yearly PM10, longer sample
    v3_y_avg = fit(dy, ["pm10"] + V3_CONTROLS + GREEN_TERMS)
    checks = [
        ("Yearly PM10 (longer sample, 2005-2014)", v3_y),
        ("All candidate controls (incl. all road and noise bands, heritage)",
         fit(d, ["pm10"] + INTERACTIONS + ALL_CONTROLS + ["lowgreen"])),
        ("Without main roads and road noise", fit(d, drop(MAIN, NEW_DATA))),
        ("Only PM10 × house (no green terms)", fit(d, drop(MAIN, ["pm10_x_lowgreen"] + GREEN_TERMS))),
        ("Only PM10 × low green", fit(d, drop(MAIN, ["pm10_x_house"]))),
        ("+ borough-by-year fixed effects", two_fe_ols(d, MAIN, "lsoacode", "la_year")),
        ("LSOA-by-year fixed effects", fit(d, varying(d, MAIN[1:], "lsoa_year"), area_fe="lsoa_year")),
        ("Standard errors clustered by borough", fit(d, MAIN, cluster="oslaua")),
    ]
    lay, ly = checks[5][1], checks[6][1]
    v1_m8 = fit(dy, ["pm10"] + PROPERTY + ["dist_to_cbd", "tube_km", "bus_km", "Bus_stopscount_200m", "park_km",
                                           "thames_km", "heritage_count_200m", "pay_k", "jobdensity",
                                           "unemployment", "pop_k", "percent_noedu", "percent_minorities",
                                           "lone_hh"])
    v2_e = fit(d, ["pm10"] + [v for k in ["property", "far", "bus", "heritage", "census", "borough"]
                              for v in BLOCKS[k][1]])
    print("v3 models done")

    # PM10 effect and WTP for the four groups
    low = d["lowgreen"] == 1
    groups = [  # label, weights, sales in the group
        ("Flats, greener postcodes", {"pm10": 1}, (d["house"] == 0) & ~low),
        ("Flats, least green third", {"pm10": 1, "pm10_x_lowgreen": 1}, (d["house"] == 0) & low),
        ("Houses, greener postcodes", {"pm10": 1, "pm10_x_house": 1}, (d["house"] == 1) & ~low),
        ("Houses, least green third", {"pm10": 1, "pm10_x_house": 1, "pm10_x_lowgreen": 1},
         (d["house"] == 1) & low),
    ]
    g_eff = [(lab, *lincom(v3, w), d.loc[mask, "price"].mean(), int(mask.sum())) for lab, w, mask in groups]
    b_avg, s_avg = coef(v3_avg, "pm10")
    low_cut = d["green_500"].quantile(1 / 3)

    plot([("v1 (M8): yearly PM10", *coef(v1_m8, "pm10"), False),
          ("v2: 3-year average PM10", *coef(v2_e, "pm10"), False),
          ("v3: all homes", b_avg, s_avg, True)]
         + [(f"v3: {lab.lower()}", c, se, True) for lab, c, se, _, _ in g_eff]
         + [("v3: all homes, yearly PM10", *coef(v3_y_avg, "pm10"), False),
            ("v3: all candidate controls", *coef(full, "pm10"), False)])

    # ===================== Markdown =====================
    audit_rows = []
    for key, label, k, ws, W, q, p, b, shift, dbic, keep in audit:
        kept = ("**kept, simplified**" if keep and key in ["roads", "noise"] else "**kept, linear**"
                if keep and key == "bus" else "**kept**" if keep
                else "dropped as a control; main effect of the green interaction" if key == "green" else "dropped")
        audit_rows.append(f"| {label} | {k} | {ws * 100:.1f}% | {W:.1f} | {p_fmt(p)} | {pct(b)} | {shift:+.2f} | "
                          f"{dbic:+,.0f} | {kept} |")

    coef_rows = []
    for v in MAIN:
        cells = []
        for m in [v3_avg, v3, v3_y]:
            if v in m["res"].index:
                c, s_ = coef(m, v)
                cells.append(f"{c:+.5f}{stars(c, s_)} ({s_:.5f})")
            else:
                cells.append("")
        coef_rows.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")

    def rcell(m, t):
        if t not in m["res"].index:
            return "absorbed" if t == "pm10" else ""
        c, s_ = coef(m, t)
        return f"{c * 100:+.3f}{stars(c, s_)} ({s_ * 100:.3f})"

    rob_rows = [f"| {lab} | " + " | ".join(rcell(m, t) for t in ["pm10"] + INTERACTIONS) + f" | {m['n']:,} |"
                for lab, m in [("**v3 (main)**", v3)] + checks]

    def n_sig(t):
        """Specifications with the term in which it is significant at 5% with the same sign as in the main model."""
        ms = [v3] + [m for _, m in checks if t in m["res"].index]
        ok = sum(abs(coef(m, t)[0] / coef(m, t)[1]) > 1.96 and np.sign(coef(m, t)[0]) == np.sign(coef(v3, t)[0])
                 for m in ms)
        return ok, len(ms)

    sig_house, sig_green = n_sig("pm10_x_house"), n_sig("pm10_x_lowgreen")
    bonf = 0.05 / N_INTERACTIONS
    p_int = {t: p_normal(coef(v3, t)[0] / coef(v3, t)[1]) for t in INTERACTIONS}
    grp_rows = [f"| {lab} | {pct(c)}{stars(c, se)} | {se * 100:.3f} | {gbp(-c * pr)} | {gbp(pr)} | {n:,} |"
                for lab, c, se, pr, n in g_eff]
    not_needed = [a for a in audit if not a[-1] and a[0] != "green"]

    n_d = len(d)
    md = f"""# Model v3: The Complete Final Model

Generated by `model_v3.py`. The audit and all model results are produced by the script; the results of earlier
tests in the inventory are quoted from the files named there. Outcome: log sale price. Coefficients ×100 give the
% change in price.

## 1. Everything we controlled for or tested

| What | Where | Result | In v3? |
|---|---|---|---|
| **PM10, yearly** at the nearest monitoring site | `model_v1.md` | -0.21% per µg/m³ in M8 | Robustness check |
| **PM10, 3-year average** (t-2 to t) | `model_v1_pm10_smoothing.md` | Larger effect (-0.37%): yearly swings are measurement error | **Yes, main PM10 measure** |
| PM10, last year / 2SLS with last year as instrument | `model_v1_pm10_smoothing.md` | -0.20% / -0.52% | No (check only) |
| Property type, new build, tenure | `model_v1.md` | Large and significant; vary within LSOAs | **Yes** |
| Distance to centre and to the tube | `model_v1.md`, `model_v1_non-linear.md` | Key controls *without* area fixed effects (they turn the PM10 sign); mostly absorbed by LSOA FE but still improve the fit | **Yes** (log + log², see audit) |
| Distance to bus stop, bus stops within 200 m | `model_v1.md` | Vary within LSOAs | **Yes** (linear) |
| Distance to a Grade I park, to the Thames | `model_v1.md` | Mostly absorbed by LSOA FE; the Thames term picks up a riverside premium | **Yes** (with the other distances) |
| Grade I heritage buildings within 200 m | `model_v1.md` | No effect within LSOAs (audit) | No |
| Non-linear distances (log + log²) | `model_v1_non-linear.md` | Lowest BIC in every fixed-effects level; PM10 unchanged | **Yes** |
| Borough pay, job density, unemployment, population (by year) | `model_v1.md`, `model_v1_income_groups.md` | Change over time, so they can confound the PM10 changes; job density moved PM10 a lot in v1 | **Yes** (pay as a cubic) |
| Census shares: no qualifications, minorities, lone parents | `model_v1.md` | Output-area level, so they vary within LSOAs | **Yes** |
| NOx | `model_v1.md` | No effect once PM10 is in | No |
| Crime | `model_v1.md` | No effect; only 2010-2014 | No |
| Year-quarter fixed effects | `model_v1.md` | Remove the London-wide price trend | **Yes** |
| LSOA fixed effects | `model_v1.md` | Compare sales within small neighbourhoods | **Yes** |
| Borough-by-year / LSOA-by-year fixed effects | `model_v1.md`, `model_v2.md` | Strict checks | Robustness check |
| Income (LSOA, MSOA, borough) × PM10 | `model_v1_income_groups.md`, `model_v1_income_small_area.md` | No income gradient | No |
| PM10 × house | `model_v2.md`, `model_v2_roads.md` | **Robust**: flats carry the larger discount, also with LSOA-by-year FE | **Yes, interaction** |
| PM10² | `model_v2.md` | No non-linearity | No |
| PM10 × share aged 0-15 / 65+ | `model_v2.md` | No robust effect | No |
| PM10 × education | `model_v2.md` | Weak hint, not robust | No |
| Distance to a main road (OS Open Roads) | `model_v2_roads.md` | -2.0% to -2.5% within 50 m; nothing beyond | **Yes** (within 50 m) |
| Road noise (Defra 2012) | `model_v2_roads.md` | About -2.3% at 65-74.9 dB; nothing below 65 dB | **Yes** (65-74.9 dB, 75 dB+) |
| PM10 × near a main road / × noise | `model_v2_roads.md` | No effect | No |
| Green share within 500 m (OS Open Greenspace) | `model_v1_green_space.md` | +0.81% per +10 pp within LSOAs (10% level); not significant in the audit | **Yes**, as a main effect of the green interaction |
| PM10 × green space | `model_v1_green_space.md`, this file | Larger discount where there is almost no green space; in v1 not robust to LSOA-by-year FE, in v3 robust | **Yes, interaction** (least green third) |
| Local park within 400 m, × PM10 | `model_v1_green_space.md` | No effect | No |
| Energy efficiency (EPC), × PM10 | `model_v1_energy_efficiency.md` | Hypothesis not supported; the postcode score mixes building age and type; only 76% of sales matched | No |

## 2. Which controls does the final model need? (audit)

Start: the v2 controls plus all road bands, all noise bands and the green share, with the 3-year average of PM10,
LSOA and year-quarter fixed effects ({full['n']:,} sales). PM10 effect in this model: {pct(b_full)}
(s.e. {s_full * 100:.3f} pp). Each block is then dropped once.

| Block | Terms | Variance left within LSOAs | Joint test χ² | p | PM10 effect without the block | Shift (s.e.) | ΔBIC without the block | v3 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(audit_rows)}

"Variance left within LSOAs": share of the variable's variance that remains after removing LSOA means; with LSOA
fixed effects, only this part is used. ΔBIC > 0 means the fit gets worse without the block. "Shift (s.e.)": change
of the PM10 effect in standard errors of the full model.

**Rule.** A block stays if it is jointly significant at {KEEP_P:.0%} (clustered Wald test) **or** if dropping it
moves the PM10 effect by {KEEP_SHIFT} standard errors or more. A kept block is then simplified to the terms that
matter.

- **Not needed: {', '.join(BLOCKS[a[0]][0] for a in not_needed)}.**
  {' '.join(f'{BLOCKS[a[0]][0]}: p {p_fmt(a[6])}, PM10 shift {a[8]:+.2f} s.e.' for a in not_needed)}
- **Green share: not needed as a control, but part of the green interaction.** On its own it raises prices only a
  little within LSOAs ({pct(coef(full, 'green10')[0])} per +10 pp, p {p_fmt(A['green'][6])}). It returns in v3 as a
  main effect of PM10 × low green (section 3), together with the low-green dummy.
- **Needed because it changes the PM10 effect: the borough economy.** It changes over time like PM10; without it
  the effect would be {pct(A['borough'][7])} instead of {pct(b_full)} ({A['borough'][8]:+.2f} s.e.). Property type also
  moves it ({A['property'][8]:+.2f} s.e.).
- **Needed because they explain prices within LSOAs:** bus access, census shares, main roads, road noise. They
  hardly move the PM10 effect (PM10 is identified from changes over time, these are fixed), but they are real price
  determinants and improve precision. The census shares also protect the road and noise estimates against poorer
  streets being nearer main roads.
- **Distances to the centre, tube, Grade I park and Thames** keep only {A['far'][3] * 100:.1f}% of their variance
  within LSOAs, yet they are jointly significant (χ² = {A['far'][4]:.1f}) and the fit is worse without them
  (ΔBIC {A['far'][9]:+,.0f}; mostly the riverside premium in the Thames term, `model_v1_non-linear.md`). They stay,
  in log + log² form; their coefficients should not be interpreted.
- **Simplified.** Only homes within 50 m of a main road and noise of 65 dB or more are discounted; the other bands
  (50-400 m, 55-64.9 dB) are jointly zero (χ² = {W_rest:.1f}, df {q_rest}, p {p_fmt(p_rest)}), so v3 keeps three
  dummies. Bus distance enters linearly (its log-quadratic form added nothing, `model_v1_non-linear.md`).

## 3. The v3 model

ln(price) = β · PM10₃ + θ₁ · PM10₃ × house + θ₂ · PM10₃ × low green + property + distances + bus access
+ census shares + borough economy + main road + road noise + green share + low green + LSOA FE
+ year-quarter FE + error

PM10₃ = mean PM10 of the sale year and the two years before. "Low green" = the least green third of postcodes
(below {low_cut * 100:.1f}% green space within 500 m). The green share and the low-green dummy enter as main
effects, so that PM10 × low green compares how prices react to the same change in PM10, not price levels.
Standard errors clustered by MSOA. Sample: {n_d:,} sales, 2007-2014 ({n_missing} sales without road or green
values).

### Headline

- **PM10 effect, all homes: {pct(b_avg)} per µg/m³** (s.e. {s_avg * 100:.3f} pp), about
  **{gbp(-b_avg * mean_price)} per home** for 1 µg/m³ less PM10 at the mean price ({gbp(mean_price)}).
  Elasticity at mean PM10: {b_avg * d['pm10'].mean():.3f}.
- **PM10 × house: {ctxt(v3, 'pm10_x_house')}**: houses lose less value from PM10 than flats.
- **PM10 × low green: {ctxt(v3, 'pm10_x_lowgreen')}**: homes with almost no green space nearby lose more.
- **Additional data:** within 50 m of a main road {price_pct(coef(v3, 'road_0_50')[0])}, road noise 65-74.9 dB
  {price_pct(coef(v3, 'noise_65_75')[0])}, 75 dB or more {price_pct(coef(v3, 'noise_75')[0])} (few sales); green
  space enters through the interaction.

**PM10 effect by group** (from the main model):

| Group | PM10 effect | Std. error (pp) | WTP per µg/m³ | Mean price | Sales |
|---|---|---|---|---|---|
{chr(10).join(grp_rows)}

WTP = −effect × mean price of the group.

![PM10 effect across model versions](../figures/v3_final.png)

### Full coefficient table

| Variable | v3, average PM10 effect | **v3 (main)** | v3, yearly PM10 |
|---|---|---|---|
{chr(10).join(coef_rows)}
| Year-quarter FE | Yes | Yes | Yes |
| LSOA FE | Yes | Yes | Yes |
| Sales | {v3_avg['n']:,} | {v3['n']:,} | {v3_y['n']:,} |
| R² | {v3_avg['r2']:.3f} | {v3['r2']:.3f} | {v3_y['r2']:.3f} |

Coefficients (standard errors clustered by MSOA). \\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01. In the main
model the `pm10` row is the effect for flats in greener postcodes. The distances keep little variation within
LSOAs, the census shares hardly vary within LSOAs and the borough terms are controls for time-varying confounders;
their coefficients should not be interpreted.

### Robustness

| Specification | PM10 (flats, greener) | PM10 × house | PM10 × low green | Sales |
|---|---|---|---|---|
{chr(10).join(rob_rows)}

Coefficient ×100 (standard error ×100). Empty cell: term not in that specification. LSOA-by-year fixed effects
compare homes sold in the same LSOA in the same year; PM10 itself is absorbed, both interactions are still
identified.

- **PM10 × house** keeps its sign and is significant at 5% in {sig_house[0]} of {sig_house[1]} specifications.
- **PM10 × low green** keeps its sign and is significant at 5% in {sig_green[0]} of {sig_green[1]} specifications.
- The two interactions hardly affect each other: compare the two "Only ..." rows with the main row.

**A caution on PM10 × low green.** About {N_INTERACTIONS} PM10 interactions were tested across the project. With a
Bonferroni correction for that many tests (p < {bonf:.4f}), PM10 × house
{'passes' if p_int['pm10_x_house'] < bonf else 'does not pass'} (p {p_fmt(p_int['pm10_x_house'])}) and PM10 × low green
{'passes' if p_int['pm10_x_lowgreen'] < bonf else 'does not pass'} (p {p_fmt(p_int['pm10_x_lowgreen'])}). The low-green
cut (least green third) was also chosen after seeing the tercile results in `model_v1_green_space.md`, where the
interaction failed with LSOA-by-year fixed effects (yearly PM10, v1 controls). It is robust within this model, but
it should be presented as the less certain of the two interactions.

## 4. Verdict

- **The PM10 effect hardly depends on the choice of controls, except one block.** With all candidate controls it
  is {pct(b_full)}; v3 gives {pct(b_avg)} for all homes. Only the borough economy moves it noticeably (without it
  {pct(A['borough'][7])}), which is why it stays. What matters most for its size is the PM10 measure (3-year
  average vs yearly: {pct(b_avg)} vs {pct(coef(v3_y_avg, 'pm10')[0])}).
- **Borough-by-year fixed effects.** In v1 these took the PM10 effect to zero; in v3 the effect for flats in
  greener postcodes is {pct(coef(lay, 'pm10')[0])} ({sig_txt(*coef(lay, 'pm10'))}), and both interactions stay
  significant.
- **The discount is concentrated in flats and in places with little green space.** The PM10 effect is
  {pct(g_eff[1][1])} per µg/m³ for flats in the least green third and {pct(g_eff[2][1])} for houses in greener
  postcodes. Both interactions hold with
  LSOA-by-year fixed effects (house {ctxt(ly, 'pm10_x_house')}; low green {ctxt(ly, 'pm10_x_lowgreen')}).
- **The additional data matter.** Main roads and noise shift prices between postcodes of the same LSOA; green
  space changes how much PM10 matters.
- **What we leave out and why:** heritage (no effect within LSOAs), NOx and crime (no effect), PM10² and PM10 ×
  income / age / education / roads / noise / efficiency (no robust effect), energy efficiency (not the home's own
  rating, 24% of sales lost).

## Limitations

- **PM10 from monitoring sites.** The 3-year average reduces measurement error but PM10 is still the value at the
  nearest of about 60 to 74 sites, so the estimate is probably still biased towards zero.
- **Postcode centroids.** Roads, noise and green space are measured from the postcode centre, not the home.
- **Data years.** Roads and green space are current releases and noise is from 2012, while the sales run
  from 2007 to 2014.
- **Not causal.** v3 is the best-controlled association we can estimate with these data; a causal willingness
  to pay would need a quasi-experiment, such as a policy that changed pollution in some areas but not others.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE}")


if __name__ == "__main__":
    main()
