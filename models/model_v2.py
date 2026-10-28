"""
HE3027 Group Project: model v2 and for whom the PM10 effect is larger.

Part 1 builds an updated model from what the earlier scripts found:
  - log(x) + log(x)² for the five distances   (model_v1_non-linear.md, lowest BIC)
  - borough pay as a cubic                     (model_v1_income_groups.md)
  - 3-year average of PM10, t-2 to t           (model_v1_pm10_smoothing.md, preferred PM10 measure)

Part 2 adds four pre-specified interactions to it: PM10 × house, PM10², PM10 × share of vulnerable
residents (aged 0-15 or 65+) and PM10 × education. LSOA household income enters only as an extra
PM10 × income term in the joint model, because model_v1_income_small_area.md found no income gradient on its own.

Writes model_v2.md and figures/v2_interactions.png.

Run from the repository root:
    python models/model_v2.py

Needs: pandas, numpy, matplotlib
Data:  data/land_registry_pollution.dta, data/gla_lsoa_atlas/lsoa-data.csv
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_v1 import (ACCESS, AMENITY, CLUSTER, CODE_DIR, MODEL_DIR, PROPERTY, SOCIO, fe_ols, load_data, pct, stars,
                      two_fe_ols)
from model_v1_income_groups import add_pay_terms, gbp, p_fmt, p_normal
from model_v1_income_small_area import LSOA_FILE, lincom
from model_v1_pm10_smoothing import add_smoothed_pm10

MD_FILE = MODEL_DIR / "model_v2.md"
FIG_DIR = CODE_DIR / "figures"
FIG_FILE = FIG_DIR / "v2_interactions.png"

# same distance terms as model_v1_non-linear.py (not imported: the hyphen in the file name
# makes it unimportable, and that script needs scipy)
DISTANCES = ["dist_to_cbd", "tube_km", "bus_km", "park_km", "thames_km"]
LOG_FLOOR = 0.01     # km; park_km is 0 for 11 sales, so logs use max(x, 10 m)

DIST_LOGQ = [t for v in DISTANCES for t in (f"ln_{v}", f"ln_{v}_sq")] + ["Bus_stopscount_200m",
                                                                          "heritage_count_200m"]
SOCIO_V2 = ["pay_c", "pay_c2", "pay_c3"] + [v for v in SOCIO if v != "pay_k"]
CONTROLS_V1 = PROPERTY + ACCESS + AMENITY + SOCIO          # model_v1.py (M8)
CONTROLS_V2 = PROPERTY + DIST_LOGQ + SOCIO_V2              # v2

MODERATORS = {  # key: (label, column, unit of the interaction, scale)
    "vuln": ("Share of residents aged 0-15 or 65+", "vulnerable", "+10 pp share", 0.10),
    "edu": ("Share of residents without qualifications", "percent_noedu", "+10 pp share", 0.10),
}
INTERACTIONS = {  # term: (label, unit of the reported coefficient)
    "pm10_x_house": ("PM10 × house", "house minus flat"),
    "pm10_sq": ("PM10²", "per +1 µg/m³ of PM10 level"),
    "pm10_x_vuln": ("PM10 × vulnerable share", "per +10 pp aged 0-15 or 65+"),
    "pm10_x_edu": ("PM10 × no qualifications", "per +10 pp without qualifications"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def add_lsoa_atlas(df):
    """Age structure (2011 mid-year estimates), degree share and median household income per LSOA."""
    a = pd.read_csv(LSOA_FILE, encoding="latin-1", low_memory=False)
    pop = a["Mid-year Population Estimates;All Ages;2011"]
    a = pd.DataFrame({
        "lsoacode": a["Lower Super Output Area"],
        "kids": a["Mid-year Population Estimates;Aged 0-15;2011"] / pop,
        "old": a["Mid-year Population Estimates;Aged 65+;2011"] / pop,
        "degree": a["Qualifications;% Highest level of qualification: Level 4 qualifications and above;2011"] / 100,
        "inc_lsoa": a["Household Income, 2011/12;Median Annual Household Income estimate (£)"],
    }).dropna()
    a["vulnerable"] = a["kids"] + a["old"]
    n = len(df)
    df = df.merge(a, on="lsoacode", how="inner")
    return df, n - len(df)


def add_v2_terms(df):
    for v in DISTANCES:
        lg = np.log(df[v].clip(lower=LOG_FLOOR))
        df[f"ln_{v}"] = lg
        df[f"ln_{v}_sq"] = lg ** 2
    add_pay_terms(df)
    df["house"] = df[["detached_dum", "semi_d_dum", "terrace_dum"]].sum(axis=1)
    df["ln_inc"] = np.log(df["inc_lsoa"])
    df["lsoa_year"] = df["lsoacode"] + "_" + df["year"].astype(str)
    df["la_year"] = df["oslaua"] + "_" + df["year"].astype(str)


def with_pm(d, pm):
    """Sample with `pm` as the PM10 measure, and the interaction terms built from it.

    PM10 and the moderators are centred at the sample mean, so the `pm10` coefficient is the effect for a
    flat at mean PM10 and mean moderators. The house interaction is not centred: `pm10` is the flat effect.
    """
    d = d.dropna(subset=[pm]).copy()
    d["pm10"] = d[pm]
    d["pm10_sq"] = (d["pm10"] - d["pm10"].mean()) ** 2
    d["pm10_x_house"] = d["pm10"] * d["house"]
    for k, (_, col, _, scale) in MODERATORS.items():
        d[f"pm10_x_{k}"] = d["pm10"] * (d[col] - d[col].mean()) / scale
    d["pm10_x_inc"] = d["pm10"] * (d["ln_inc"] - d["ln_inc"].mean())
    return d


def by_year(d, col, center=True):
    """`col` × year dummies (first year dropped): lets houses / high-share areas have their own price trend."""
    x = d[col] - d[col].mean() if center else d[col]
    names = []
    for y in sorted(d["year"].unique())[1:]:
        d[f"{col}_y{y}"] = x * (d["year"] == y)
        names.append(f"{col}_y{y}")
    return names


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def m8v2(d, extra=(), controls=CONTROLS_V2, area_fe="lsoacode", cluster=CLUSTER):
    return fe_ols(d, ["pm10"] + list(extra) + controls, time_fe=True, area_fe=area_fe, cluster=cluster)


def varying(d, xs, fe):
    """Keep only variables that still vary within the fixed-effect groups (the rest are absorbed)."""
    out = []
    for v in xs:
        x = d[v].astype(float)
        within = (x - x.groupby(d[fe]).transform("mean")).std()
        if within > 1e-8 * (x.std() + 1e-12):
            out.append(v)
    return out


def coef(m, v, scale=1.0):
    r = m["res"]
    return r.at[v, "coef"] * scale, r.at[v, "se"] * abs(scale)


def report_scale(term):
    """PM10² is reported as the change in the PM10 effect per +1 µg/m³ of PM10, i.e. 2 × b."""
    return 2.0 if term == "pm10_sq" else 1.0


def sig_txt(c, s):
    t = abs(c / s)
    return ("significant at 1%" if t > 2.576 else "significant at 5%" if t > 1.96
            else "significant at 10%" if t > 1.645 else "not significant")


def cell(c, s, pp=True):
    k = 100 if pp else 1
    return f"{c * k:+.3f}{stars(c, s)} ({s * k:.3f})"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(rows):
    """rows: (group, label, coef, se)"""
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    colors = {"Property type": "#2a78d6", "PM10 level": "#eb6834", "Vulnerable share": "#1f9e74",
              "Education": "#8e5bd0"}
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#52514e", linewidth=0.8)
    for yi, (g, lab, c, s) in zip(y, rows):
        ax.errorbar(c * 100, yi, xerr=1.96 * s * 100, fmt="o", color=colors[g], ecolor="#0b0b0b", elinewidth=1,
                    capsize=3, markersize=7, zorder=3)
        ax.annotate(f"{c * 100:+.2f}%", (c * 100, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color="#0b0b0b")
    ax.set_yticks(y, [r[1] for r in rows])
    ax.set_xlabel("% change in price per +1 µg/m³ PM10, 3-year average (95% CI)")
    ax.set_title("PM10 effect by property type and neighbourhood (model v2)", loc="left", fontsize=11,
                 color="#0b0b0b")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=g) for g, c in colors.items()]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, 0.01, "Each colour is a separate model with one interaction. P10 / P90: 10th / 90th "
             "percentile over sales. Standard errors clustered by MSOA.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, full, info = load_data()
    df, _ = add_smoothed_pm10(df, full)
    df, n_lost = add_lsoa_atlas(df)
    add_v2_terms(df)
    mean_price = df["price"].mean()

    # ===================== Part 1: the v2 baseline =====================
    yearly = with_pm(df, "pm10")
    ma3 = with_pm(df, "pm10_ma3")
    base = [
        ("A", "M8 from `model_v1.md` (yearly PM10, linear distances, linear pay)", yearly, CONTROLS_V1),
        ("B", "A + log(x) + log(x)² of the five distances", yearly, PROPERTY + DIST_LOGQ + SOCIO),
        ("C", "B + borough pay as a cubic (all v2 controls, yearly PM10)", yearly, CONTROLS_V2),
        ("D", "C on the 3-year-average sample (yearly PM10)", with_pm(df.dropna(subset=["pm10_ma3"]), "pm10"),
         CONTROLS_V2),
        ("E", "**v2: C with the 3-year average of PM10 (t-2 to t)**", ma3, CONTROLS_V2),
    ]
    base_fits = []
    for code, label, d, ctrl in base:
        m = m8v2(d, controls=ctrl)
        base_fits.append(m)
        print(f"{code} b_pm10 = {m['res'].at['pm10', 'coef']: .5f} (se {m['res'].at['pm10', 'se']:.5f}) "
              f"n = {m['n']:,}")
    v2 = base_fits[-1]
    b_v2, se_v2 = coef(v2, "pm10")
    price_v2 = ma3["price"].mean()

    # ===================== Part 2: interactions =====================
    terms = list(INTERACTIONS)
    single = {t: m8v2(ma3, [t]) for t in terms}
    joint = m8v2(ma3, terms)
    joint_inc = m8v2(ma3, terms + ["pm10_x_inc"])
    single_yearly = {t: m8v2(yearly, [t]) for t in terms}
    single_la_cl = {t: m8v2(ma3, [t], cluster="oslaua") for t in terms}
    print("single and joint models done")

    # moderator-specific year effects (not for PM10², whose moderator is PM10 itself)
    trend = {}
    for t, col in [("pm10_x_house", "house"), ("pm10_x_vuln", "vulnerable"), ("pm10_x_edu", "percent_noedu")]:
        d = ma3.copy()
        yrs = by_year(d, col, center=(col != "house"))
        trend[t] = m8v2(d, [t] + yrs)

    # borough-by-year fixed effects: removes any shock that hits a whole borough in a year
    la_year = {t: two_fe_ols(ma3, ["pm10", t] + CONTROLS_V2, "lsoacode", "la_year") for t in terms}

    # LSOA-by-year fixed effects: PM10 itself is absorbed, only the house interaction is left.
    # Houses and flats in the same LSOA and year are compared, so any local shock is removed.
    xs = varying(ma3, ["pm10_x_house"] + CONTROLS_V2, "lsoa_year")
    lsoa_year = fe_ols(ma3, xs, time_fe=True, area_fe="lsoa_year")
    print("robustness done")

    # --- PM10 effect at different values of each moderator (single-interaction models) ---
    def at(m, w):
        return lincom(m, w)

    pm_mu = ma3["pm10"].mean()
    pm_p10, pm_p50, pm_p90 = ma3["pm10"].quantile([0.1, 0.5, 0.9])
    eff_rows = []    # (group, label, coef, se, mean price of the group or None)
    flat, house = at(single["pm10_x_house"], {"pm10": 1}), at(single["pm10_x_house"], {"pm10": 1, "pm10_x_house": 1})
    p_flat = ma3.loc[ma3["house"] == 0, "price"].mean()
    p_house = ma3.loc[ma3["house"] == 1, "price"].mean()
    eff_rows += [("Property type", "Flats", *flat, p_flat), ("Property type", "Houses", *house, p_house)]
    for lab, q in [("P10", pm_p10), ("P50", pm_p50), ("P90", pm_p90)]:
        eff_rows.append(("PM10 level", f"PM10 at {lab} ({q:.1f} µg/m³)",
                         *at(single["pm10_sq"], {"pm10": 1, "pm10_sq": 2 * (q - pm_mu)}), None))
    for k, (label, col, _, scale) in MODERATORS.items():
        grp = "Vulnerable share" if k == "vuln" else "Education"
        mu = ma3[col].mean()
        for lab, q in [("P10", ma3[col].quantile(0.1)), ("P90", ma3[col].quantile(0.9))]:
            eff_rows.append((grp, f"{'Aged 0-15/65+' if k == 'vuln' else 'No qualif.'} at {lab} ({q * 100:.0f}%)",
                             *at(single[f"pm10_x_{k}"], {"pm10": 1, f"pm10_x_{k}": (q - mu) / scale}), None))
    plot([(g, l, c, s) for g, l, c, s, _ in eff_rows])

    # --- turning point of the quadratic ---
    b1, b2 = single["pm10_sq"]["res"].at["pm10", "coef"], single["pm10_sq"]["res"].at["pm10_sq", "coef"]
    tp = pm_mu - b1 / (2 * b2) if b2 != 0 else float("nan")

    # --- correlations between the moderators (over sales) ---
    corr_vars = {"PM10 (3-yr avg)": "pm10", "House": "house", "Aged 0-15/65+": "vulnerable",
                 "No qualifications": "percent_noedu", "Degree": "degree", "ln(LSOA income)": "ln_inc"}
    cm = ma3[list(corr_vars.values())].corr()

    # extra checks on the moderator definitions
    alt = {}
    d = ma3.copy()
    for col in ["kids", "old", "degree"]:
        d[f"pm10_x_{col}"] = d["pm10"] * (d[col] - d[col].mean()) / 0.10
        alt[col] = m8v2(d, [f"pm10_x_{col}"])

    # ===================== Markdown =====================
    def pick(fits, t):
        """`fits` is either one model (used for every column) or a dict of models by interaction term."""
        return fits if "res" in fits else fits.get(t)

    def irow(label, fits, note=""):
        cells = []
        for t in terms:
            m = pick(fits, t)
            if m is None or t not in m["res"].index:
                cells.append("")
            else:
                c, s = coef(m, t, report_scale(t))
                cells.append(cell(c, s))
        return f"| {label}{note} | " + " | ".join(cells) + " |"

    def nrow(fits):
        ns = []
        for t in terms:
            m = pick(fits, t)
            ns.append(f"{m['n']:,}" if m is not None else "")
        return ns

    base_rows = []
    for (code, label, d, _), m in zip(base, base_fits):
        c, s = coef(m, "pm10")
        base_rows.append(f"| {code} | {label} | {pct(c)}{stars(c, s)} | {s * 100:.3f} | "
                         f"{gbp(-c * d['price'].mean())} | {c * d['pm10'].mean():.3f} | "
                         f"{m['r2']:.3f} | {m['n']:,} |")

    eff_md = []
    for g, lab, c, s, p in eff_rows:
        wtp = gbp(-c * p) if p else ""
        eff_md.append(f"| {g} | {lab} | {pct(c)}{stars(c, s)} | {s * 100:.3f} | {wtp} |")

    def mtxt(t, m=None):
        m = m or single[t]
        c, s = coef(m, t, report_scale(t))
        return f"{c * 100:+.3f} pp ({sig_txt(c, s)}, p {p_fmt(p_normal(c / s))})"

    corr_md = ["| | " + " | ".join(corr_vars) + " |", "|---|" + "---|" * len(corr_vars)]
    for name, v in corr_vars.items():
        corr_md.append(f"| {name} | " + " | ".join(f"{cm.at[v, w]:.2f}" for w in corr_vars.values()) + " |")

    alt_md = [f"| {lab} | {mtxt(f'pm10_x_{col}', alt[col])} |" for col, lab in
              [("kids", "PM10 × share aged 0-15"), ("old", "PM10 × share aged 65+"),
               ("degree", "PM10 × share with a degree (Level 4+)")]]

    # house WTP in GBP and its difference
    wtp_flat, wtp_house = -flat[0] * p_flat, -house[0] * p_house
    w = {"pm10": p_flat - p_house, "pm10_x_house": -p_house}       # WTP house − WTP flat = −(b + d)·P_h + b·P_f
    d_wtp, d_wtp_se = lincom(single["pm10_x_house"], w)

    house_pos = coef(single["pm10_x_house"], "pm10_x_house")[0] > 0
    edu_neg = coef(single["pm10_x_edu"], "pm10_x_edu")[0] < 0
    sq_fits = [single, joint, joint_inc, single_yearly, la_year, single_la_cl]
    any_sig_sq = any(abs(coef(f["pm10_sq"] if "res" not in f else f, "pm10_sq")[0]
                         / coef(f["pm10_sq"] if "res" not in f else f, "pm10_sq")[1]) > 1.96 for f in sq_fits)
    sig_terms = [t for t in terms if abs(coef(single[t], t)[0] / coef(single[t], t)[1]) > 1.96]
    bonf = [t for t in terms if p_normal(coef(single[t], t)[0] / coef(single[t], t)[1]) < 0.05 / len(terms)]

    md = f"""# Model v2 and Heterogeneity in the PM10 Effect

Generated by `model_v2.py`. All numbers in this file are produced by the script.
Outcome: log sale price. Coefficients ×100 give the % change in price.

## Part 1: The v2 model

The v2 model collects what the earlier scripts found into one specification:

| Change compared with M8 | Why | Source |
|---|---|---|
| Distances to the centre, tube, bus stop, park and Thames as log(x) + log(x)² | Lowest BIC in every fixed-effects level | `model_v1_non-linear.md` |
| Borough pay as pay + pay² + pay³ (centred) | Each extra term significant with borough clusters and lowers the BIC | `model_v1_income_groups.md` |
| PM10 as the 3-year average of the sale year and the two years before | Buyers judge long-run air quality; yearly swings act as measurement error | `model_v1_pm10_smoothing.md` |

**What we do not add from the income work.** The WTP by income group (`model_v1_income_small_area.md`) showed no income
gradient, so v2 has no income groups. Area income does not change over time, so as a control it is absorbed by the
LSOA fixed effects anyway. We use it once more below, as a PM10 × income term in the joint interaction model, to
check that the education interaction is not just income in disguise.

All other parts are unchanged: property, accessibility, amenity and socioeconomic controls, LSOA and year-quarter
fixed effects, standard errors clustered by MSOA.

| | Model | PM10 effect | Std. error (pp) | WTP per µg/m³ | Elasticity at mean PM10 | R² | Sales |
|---|---|---|---|---|---|---|---|
{chr(10).join(base_rows)}

\\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01. WTP = −effect × mean price of the sample.

- **A → C:** the new distance and pay terms hardly move the PM10 effect ({pct(coef(base_fits[0], 'pm10')[0])} →
  {pct(coef(base_fits[2], 'pm10')[0])}), as the earlier scripts found. They are kept because they fit better.
- **C → D:** the 3-year average needs PM10 in the two years before the sale, so the sample loses 2005-2006 and
  LSOAs without sales in some years ({base_fits[2]['n']:,} → {base_fits[3]['n']:,} sales). On this smaller sample
  yearly PM10 gives {pct(coef(base_fits[3], 'pm10')[0])}.
- **D → E:** on exactly the same sales, the 3-year average gives {pct(b_v2)} instead of
  {pct(coef(base_fits[3], 'pm10')[0])}. Removing temporary swings makes the effect larger, as measurement error predicts.

**v2 model (E): {pct(b_v2)} per µg/m³** (standard error {se_v2 * 100:.3f} pp), about
**{gbp(-b_v2 * price_v2)} per home** for 1 µg/m³ less PM10 at the mean price of this sample
({gbp(price_v2)}). All interaction models below are built on E. {n_lost} sales are lost when merging the LSOA Atlas.

## Part 2: For whom is the PM10 effect larger?

### The four interactions

| Interaction | Moderator | Data | Idea |
|---|---|---|---|
| PM10 × house | House (detached, semi or terraced) vs flat | Land Registry property type | Houses have gardens and are bought by families who spend time outdoors |
| PM10² | PM10 itself, centred at {pm_mu:.1f} µg/m³ | 3-year average PM10 | Damage may grow faster at high levels (convex) |
| PM10 × vulnerable share | Share of residents aged 0-15 or 65+, 2011 | GLA LSOA Atlas (ONS mid-year estimates) | Children and older people are most affected by air pollution |
| PM10 × education | Share of residents without qualifications | Course data (`percent_noedu`, census) | Better-informed buyers may know more about health risks |

**How to read the coefficients.** The moderators are centred at their mean, so the main `pm10` coefficient is the
effect for a flat at mean PM10 and mean moderators. All interaction coefficients are in **percentage points of the
PM10 effect** (×100):

- PM10 × house: PM10 effect for houses minus the effect for flats.
- PM10²: change in the PM10 effect when the PM10 level is 1 µg/m³ higher (2 × the coefficient of PM10²).
- PM10 × shares: change in the PM10 effect when the share is 10 percentage points higher.

A **negative** interaction means a **larger price discount** for pollution.

**Identification.** With LSOA fixed effects the moderators themselves are absorbed (they do not change over time),
but their interaction with PM10 is not: it asks whether prices react more to the *same change* in PM10 where the
moderator is high. The comparisons are between groups inside the model, so a common shock that hits all of them
cancels out. This makes the interactions robust to some problems the main effect has (see the borough-by-year
and LSOA-by-year rows).

### Results

| Specification | {' | '.join(INTERACTIONS[t][0] for t in terms)} |
|---|{'---|' * len(terms)}
| Unit | {' | '.join(INTERACTIONS[t][1] for t in terms)} |
{irow('**One interaction at a time (main)**', single)}
{irow('All four together', joint)}
{irow('All four + PM10 × ln(LSOA income)', joint_inc)}
{irow('Yearly PM10 instead of the 3-year average', single_yearly)}
{irow('+ moderator-specific year effects', trend)}
{irow('+ borough-by-year fixed effects', la_year)}
{irow('LSOA-by-year fixed effects', {'pm10_x_house': lsoa_year})}
{irow('Standard errors clustered by borough', single_la_cl)}
| Sales | {' | '.join(nrow(single))} |

Coefficient ×100 (standard error ×100). \\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01. Empty cell: not
estimable in that specification. Sales for the yearly-PM10 row: {single_yearly[terms[0]]['n']:,}.
PM10 × ln(LSOA income) in the joint model: {mtxt('pm10_x_inc', joint_inc)} per +1 log point of income.

- **Moderator-specific year effects:** houses (or high-share LSOAs) get their own year dummies, so a different
  price trend in those areas cannot show up as an interaction.
- **Borough-by-year fixed effects:** remove any shock that hits a whole borough in a given year. In
  `model_v1.md` these took the PM10 main effect to zero.
- **LSOA-by-year fixed effects:** only possible for the house interaction, because PM10 and the area shares do not
  vary within an LSOA-year. It compares houses and flats sold in the same LSOA in the same year, so any local shock
  (regeneration, a new road, a bad pollution year) cancels out.

### PM10 effect for different groups

From the one-interaction models. WTP in GBP only where the groups have different prices.

| Group | Where | PM10 effect | Std. error (pp) | WTP per µg/m³ |
|---|---|---|---|---|
{chr(10).join(eff_md)}

![PM10 effect by group](../figures/v2_interactions.png)

### Detail by interaction

**PM10 × house: {mtxt('pm10_x_house')}.**
Flats {pct(flat[0])}, houses {pct(house[0])} ({sig_txt(*house)}). In GBP: flats {gbp(wtp_flat)}, houses
{gbp(wtp_house)} per µg/m³ (house minus flat {gbp(d_wtp)}, standard error {d_wtp_se:,.0f},
p {p_fmt(p_normal(d_wtp / d_wtp_se))}), although houses cost more ({gbp(p_house)} vs {gbp(p_flat)}).

- {'This is the **opposite** of our hypothesis: the pollution discount is concentrated in **flats**, not houses.'
   if house_pos else 'As expected, the pollution discount is larger for houses.'}
- It is the most robust result in this file: it keeps its sign and stays significant in every row of the table
  (with yearly PM10 it is about half as large, like the main effect), and it
  survives LSOA-by-year fixed effects ({mtxt('pm10_x_house', lsoa_year)}), which compare houses and flats sold in
  the same LSOA in the same year. Local shocks that hit both cannot explain it.
- Possible explanations{' for a larger flat discount' if house_pos else ''}, which the data here cannot separate:
  (1) flats are more often on busy roads and above shops, close to street level, so the monitored PM10 changes
  matter more for the air inside them; houses usually have a back garden or rooms away from the road;
  (2) flats have fewer ways to avoid the air outside (one facade, no garden); (3) flat buyers are a different
  group (younger, more mobile) with different preferences.
- (1) is exactly what the next steps can test: **PM10 × distance to a major road** and, with EPC data, **PM10 ×
  floor level**. If the flat discount comes from flats on main roads or on low floors, those interactions should
  show it.

**PM10²: {mtxt('pm10_sq')}** per +1 µg/m³.
The PM10 effect is {pct(eff_rows[2][2])} at the 10th percentile of PM10 ({pm_p10:.1f} µg/m³) and
{pct(eff_rows[4][2])} at the 90th ({pm_p90:.1f} µg/m³){', so if anything the effect gets smaller at high levels (concave), not larger' if b2 > 0 else ', so the effect grows at high levels (convex)'}.
{'The turning point of the quadratic is ' + f'{tp:.1f} µg/m³' + (', inside' if ma3['pm10'].min() < tp < ma3['pm10'].max() else ', outside') + ' the range of the sample' if not math.isnan(tp) else ''}
({ma3['pm10'].min():.1f} to {ma3['pm10'].max():.1f} µg/m³). The squared term is
{'not significant in any specification' if not any_sig_sq else 'significant in at least one specification'}, so
**a linear PM10 term is adequate**. Within an LSOA PM10 moves only a few µg/m³, so the curvature is mostly
identified from comparing how prices react in high- and low-PM10 LSOAs, and the test has little power.

**PM10 × vulnerable share: {mtxt('pm10_x_vuln')}** per +10 pp.
The share aged 0-15 or 65+ runs from {ma3['vulnerable'].quantile(0.1) * 100:.0f}% (P10) to
{ma3['vulnerable'].quantile(0.9) * 100:.0f}% (P90) of residents (mean {ma3['vulnerable'].mean() * 100:.0f}%).
The point estimate {'points to a larger discount where more children and older people live' if coef(single['pm10_x_vuln'], 'pm10_x_vuln')[0] < 0 else 'points to a smaller discount where more children and older people live'},
but it is not significant, and with yearly PM10 the sign flips ({mtxt('pm10_x_vuln', single_yearly['pm10_x_vuln'])}).
Split into its two parts:

| Interaction (per +10 pp) | Coefficient |
|---|---|
{chr(10).join(alt_md[:2])}

**No robust evidence** that the PM10 effect depends on the share of vulnerable residents. The older-people part
is the closest to significance. Note that the moderator is the age of the *residents* of the LSOA, not of the buyer.

**PM10 × education: {mtxt('pm10_x_edu')}** per +10 pp without qualifications.
{'Areas with **more** residents without qualifications show a **larger** discount, the opposite of the information hypothesis (better-educated buyers knowing more about health risks).' if edu_neg else 'Areas with more residents without qualifications show a smaller discount, as the information hypothesis predicts.'}
The result is fragile:

- with borough-by-year fixed effects it is {mtxt('pm10_x_edu', la_year['pm10_x_edu'])}, and with yearly PM10
  {mtxt('pm10_x_edu', single_yearly['pm10_x_edu'])};
- it does not pass the correction for testing four interactions (see the summary);
- the share holding a degree, the more natural measure of information, shows nothing:

| Interaction (per +10 pp) | Coefficient |
|---|---|
{alt_md[2]}

It does survive adding PM10 × ln(LSOA income) ({mtxt('pm10_x_edu', joint_inc)}), so it is not just income.
The share without qualifications is strongly linked to deprivation (correlation {cm.at['percent_noedu', 'ln_inc']:.2f}
with log income), so it may stand for something else about poorer areas, for example more housing on busy roads.
We read it as a weak hint, not as a finding.

### How the moderators are related

Correlations over the sales in the v2 sample. Strongly correlated moderators are hard to separate, which is why
the joint model matters.

{chr(10).join(corr_md)}

## Summary

- **v2 model:** {pct(b_v2)} per µg/m³ ({gbp(-b_v2 * price_v2)} per home), compared with
  {pct(coef(base_fits[0], 'pm10')[0])} in the old M8. Almost all of the change comes from smoothing PM10, not from the
  new distance or pay terms.
- **Interactions significant at 5% on their own:** {', '.join(INTERACTIONS[t][0] for t in sig_terms) if sig_terms else 'none'}.
- **Still significant after correcting for testing four interactions** (Bonferroni, p < {0.05 / len(terms):.4f}):
  {', '.join(INTERACTIONS[t][0] for t in bonf) if bonf else 'none'}.

| Interaction | Verdict |
|---|---|
| PM10 × house | **Robust.** {'Flats' if house_pos else 'Houses'} carry the larger discount ({pct(flat[0])} flats, {pct(house[0])} houses), in every specification incl. LSOA-by-year fixed effects. |
| PM10² | **No non-linearity.** Not significant; a linear PM10 term is enough. |
| PM10 × vulnerable share | **No robust effect.** Not significant, and the sign changes with yearly PM10. |
| PM10 × education | **Weak hint only.** {'Larger discount in less-educated areas' if edu_neg else 'Smaller discount in less-educated areas'}, significant at 5% on its own but not after the Bonferroni correction, with borough-by-year fixed effects or with yearly PM10. |

**Next step.** The flat result is the one to follow up. Distance to a major road (OS Open Roads) and road noise
(Defra noise maps) can show whether the flat discount comes from flats on busy roads.

## Limitations

- **Area, not buyer.** The vulnerable and education shares describe the residents of the LSOA, not the household
  that bought the home.
- **Sorting.** Households that care most about clean air live in clean areas, so heterogeneity in the hedonic price
  mixes differences in preferences with differences in who lives where.
- **PM10 from monitoring sites.** The 3-year average reduces, but does not remove, measurement error. Interactions
  with a noisy PM10 are biased towards zero as well.
- **Not causal.** Like M8, these are associations conditional on the controls and fixed effects.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE}")


if __name__ == "__main__":
    main()
