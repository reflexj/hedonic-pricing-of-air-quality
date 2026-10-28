"""
HE3027 Group Project: hedonic regressions of house prices on air pollution (PM10).

Adds the controls step by step (M1 to M8), then robustness checks (M9 NOx, crime),
and writes everything to model_v1.md.

Run from the repository root:
    python models/model_v1.py

Needs: pandas, numpy
Data:  data/land_registry_pollution.dta, data/land_registry_crime.dta (for the crime check)
"""

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent   # models/ (scripts and their .md results)
CODE_DIR = MODEL_DIR.parent                   # repository root (data/, figures/)
DATA_DIR = CODE_DIR / "data"
MD_FILE = MODEL_DIR / "model_v1.md"

TRIM = 0.01          # drop bottom and top 1% of prices
LAST_YEAR = 2014     # jobdensity is missing for all of 2015
CLUSTER = "msoa11"   # standard errors clustered by MSOA

# ---------------------------------------------------------------------------
# Variables, grouped in the order they enter the model
# ---------------------------------------------------------------------------
PROPERTY = ["detached_dum", "semi_d_dum", "terrace_dum", "newbuild", "freehold"]
ACCESS = ["dist_to_cbd", "tube_km", "bus_km", "Bus_stopscount_200m"]
AMENITY = ["park_km", "thames_km", "heritage_count_200m"]
SOCIO = ["pay_k", "jobdensity", "unemployment", "pop_k",
         "percent_noedu", "percent_minorities", "lone_hh"]

VARS = {  # name: (description, unit, source variable)
    "lnprice": ("Log of the sale price", "log GBP", "price"),
    "pm10": ("Yearly mean PM10 at the nearest monitoring site", "µg/m³", "pm10"),
    "nox": ("Yearly mean NOx at the nearest monitoring site (M9 only)", "µg/m³", "nox"),
    "detached_dum": ("Detached house (base group: flat)", "0/1", "propertyty = D"),
    "semi_d_dum": ("Semi-detached house", "0/1", "propertyty = S"),
    "terrace_dum": ("Terraced house", "0/1", "propertyty = T"),
    "newbuild": ("Newly built property", "0/1", "age = Y"),
    "freehold": ("Freehold (base group: leasehold)", "0/1", "duration = F"),
    "dist_to_cbd": ("Distance to the city centre", "km", "dist_to_cbd"),
    "tube_km": ("Distance to the nearest tube station", "km", "tube_distnear / 1000"),
    "bus_km": ("Distance to the nearest bus stop", "km", "bus_distnear / 1000"),
    "Bus_stopscount_200m": ("Bus stops within 200 m", "count", "Bus_stopscount_200m"),
    "park_km": ("Distance to the nearest Grade 1 park", "km", "Nearest_Park_dist / 1000"),
    "thames_km": ("Distance to the River Thames", "km", "thamesriv_dist / 1000"),
    "heritage_count_200m": ("Grade 1 heritage buildings within 200 m", "count", "heritage_count_200m"),
    "pay_k": ("Mean gross annual pay in the borough", "GBP 1,000", "grossannualpay / 1000"),
    "jobdensity": ("Jobs density in the borough", "jobs per resident aged 16-64 (ONS)", "jobdensity"),
    "unemployment": ("Unemployment rate in the borough", "%", "unemployment"),
    "pop_k": ("Population of the borough", "1,000 people", "totalpopulation / 1000"),
    "percent_noedu": ("Share of residents without qualifications", "share 0-1", "percent_noedu"),
    "percent_minorities": ("Share of residents from ethnic minorities", "share 0-1", "percent_minorities"),
    "lone_hh": ("Share of lone-parent households", "share 0-1", "lone_hh"),
    "log_crime": ("log(1 + all crimes in the LSOA that year) (crime check only)", "log count", "allcrime"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_stata(DATA_DIR / "land_registry_pollution.dta", convert_categoricals=False)
    info = {"n_raw": len(df)}

    lo, hi = df["price"].quantile([TRIM, 1 - TRIM])
    df = df[df["price"].between(lo, hi)].copy()
    info.update(lo=lo, hi=hi, n_trim=len(df))

    df = df[df["oslaua"].str.startswith("E09")].copy()     # 33 London boroughs only
    info["n_london"] = len(df)

    df["lnprice"] = np.log(df["price"])
    df["newbuild"] = (df["age"] == "Y").astype(int)
    df["tube_km"] = df["tube_distnear"] / 1000
    df["bus_km"] = df["bus_distnear"] / 1000
    df["park_km"] = df["Nearest_Park_dist"] / 1000
    df["thames_km"] = df["thamesriv_dist"] / 1000
    df["pay_k"] = df["grossannualpay"] / 1000
    df["pop_k"] = df["totalpopulation"] / 1000

    all_x = ["pm10", "nox"] + PROPERTY + ACCESS + AMENITY + SOCIO
    full = df.dropna(subset=["lnprice"] + [v for v in all_x if v != "jobdensity"])   # incl. 2015
    df = df[df["year"] <= LAST_YEAR].dropna(subset=["lnprice"] + all_x)
    info.update(n_final=len(df), n_lsoa=df["lsoacode"].nunique(), n_msoa=df["msoa11"].nunique(),
                n_la=df["oslaua"].nunique())
    return df, full, info


def add_crime(df):
    f = DATA_DIR / "land_registry_crime.dta"
    if not f.exists():
        return None
    c = pd.read_stata(f, columns=["lsoacode", "year", "allcrime"], convert_categoricals=False)
    c = c.groupby(["lsoacode", "year"], as_index=False).first()
    d = df.merge(c, on=["lsoacode", "year"], how="inner")
    d["log_crime"] = np.log1p(d["allcrime"])
    return d


# ---------------------------------------------------------------------------
# OLS with optional year-quarter dummies, one absorbed area fixed effect,
# and standard errors clustered by MSOA
# ---------------------------------------------------------------------------
def fe_ols(d, xvars, time_fe=False, area_fe=None, cluster=CLUSTER):
    X = d[xvars].astype(float)
    if time_fe:
        X = pd.concat([X, pd.get_dummies(d["yearquarter"], prefix="yq", drop_first=True).astype(float)], axis=1)
    y = d["lnprice"].astype(float)
    if area_fe:   # absorb the fixed effect by subtracting area means (same result as adding dummies)
        g = d[area_fe]
        X = X - X.groupby(g).transform("mean")
        y = y - y.groupby(g).transform("mean")
    else:
        X = X.assign(const=1.0)
    X = X.loc[:, X.abs().sum() > 1e-10]

    Xv, yv = X.to_numpy(), y.to_numpy()
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    b = XtX_inv @ Xv.T @ yv
    e = yv - Xv @ b
    scores = pd.DataFrame(Xv * e[:, None]).groupby(d[cluster].to_numpy()).sum().to_numpy()
    n, k, G = len(yv), Xv.shape[1], len(scores)
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((d["lnprice"] - d["lnprice"].mean()) ** 2).sum()
    res = pd.DataFrame({"coef": b, "se": se}, index=X.columns)
    V = pd.DataFrame(V, index=X.columns, columns=X.columns)
    return {"res": res, "V": V, "n": n, "k": k, "r2": r2, "time_fe": time_fe, "area_fe": area_fe}


def two_fe_ols(d, xvars, fe1, fe2, cluster=CLUSTER):
    """OLS with year-quarter dummies, fe1 absorbed (LSOA) and fe2 estimated exactly (borough-by-year).

    The LSOA effects are removed by subtracting LSOA means. The borough-by-year dummies, after the same
    demeaning, enter the normal equations through group sums and counts, so the large dummy matrix
    never has to be built. The clusters (MSOA) must contain whole LSOAs.
    """
    X = pd.concat([d[xvars].astype(float),
                   pd.get_dummies(d["yearquarter"], prefix="yq", drop_first=True).astype(float)], axis=1)
    g1 = pd.factorize(d[fe1])[0]
    g2 = pd.factorize(d[fe2])[0]
    n1, K = g1.max() + 1, g2.max() + 1

    # 1) remove fe1 (LSOA) means
    Xt = X - X.groupby(g1).transform("mean")
    yt = d["lnprice"].astype(float) - d["lnprice"].astype(float).groupby(g1).transform("mean")
    Xt = Xt.loc[:, Xt.abs().sum() > 1e-10]
    names = list(Xt.columns)
    Xv, yv = Xt.to_numpy(), yt.to_numpy()

    # 2) cross products with the demeaned fe2 dummies D~ = D - P_fe1 D
    #    D~'X~ = group sums of X~ (because X~ already has fe1 means removed)
    DX = np.column_stack([np.bincount(g2, weights=Xv[:, j], minlength=K) for j in range(Xv.shape[1])])
    Dy = np.bincount(g2, weights=yv, minlength=K)
    C = np.zeros((n1, K))
    np.add.at(C, (g1, g2), 1.0)                        # sales per (LSOA, borough-year)
    n_l = C.sum(axis=1)
    DD = np.diag(C.sum(axis=0)) - C.T @ (C / n_l[:, None])

    # 3) solve the full normal equations
    A = np.block([[Xv.T @ Xv, DX.T], [DX, DD]])
    rhs = np.concatenate([Xv.T @ yv, Dy])
    A_inv = np.linalg.pinv(A)
    beta = A_inv @ rhs
    bx, bd = beta[:Xv.shape[1]], beta[Xv.shape[1]:]
    fitted_d = bd[g2] - (C @ bd / n_l)[g1]             # D~ times bd, row by row
    e = yv - Xv @ bx - fitted_d

    # 4) clustered sandwich; residuals sum to zero within each LSOA, so D~ scores are plain group sums
    cl = pd.factorize(d[cluster])[0]
    G = cl.max() + 1
    Sx = pd.DataFrame(Xv * e[:, None]).groupby(cl).sum().to_numpy()
    Sd = np.zeros((G, K))
    np.add.at(Sd, (cl, g2), e)
    S = np.hstack([Sx, Sd])
    n, k = len(yv), A.shape[0]
    V = A_inv @ (S.T @ S) @ A_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.clip(np.diag(V)[:Xv.shape[1]], 0, None))
    res = pd.DataFrame({"coef": bx, "se": se}, index=names)
    return {"res": res, "n": n}


def stars(c, s):
    t = abs(c / s)
    return "***" if t > 2.576 else "**" if t > 1.96 else "*" if t > 1.645 else ""


def fmt(m, v):
    r = m["res"]
    if v not in r.index:
        return "", ""
    c, s = r.at[v, "coef"], r.at[v, "se"]
    return f"{c:.5f}{stars(c, s)}", f"({s:.5f})"


def table(models, names, rows):
    out = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(models)]
    for v in rows:
        cs, ss = zip(*[fmt(m, v) for m in models])
        out.append(f"| `{v}` | " + " | ".join(cs) + " |")
        out.append("| | " + " | ".join(ss) + " |")
    fe_name = {None: "No", "oslaua": "Borough", "lsoacode": "LSOA"}
    out.append("| Year-quarter FE | " + " | ".join("Yes" if m["time_fe"] else "No" for m in models) + " |")
    out.append("| Area FE | " + " | ".join(fe_name[m["area_fe"]] for m in models) + " |")
    out.append("| Observations | " + " | ".join(f"{m['n']:,}" for m in models) + " |")
    out.append("| R² | " + " | ".join(f"{m['r2']:.3f}" for m in models) + " |")
    return "\n".join(out)


def pct(b):
    return f"{b * 100:+.2f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, full, info = load_data()

    # --- main sequence ---
    steps = [
        ("M1", "PM10 only", ["pm10"], False, None),
        ("M2", "+ property characteristics", ["pm10"] + PROPERTY, False, None),
        ("M3", "+ accessibility", ["pm10"] + PROPERTY + ACCESS, False, None),
        ("M4", "+ environmental amenities", ["pm10"] + PROPERTY + ACCESS + AMENITY, False, None),
        ("M5", "+ socioeconomic controls", ["pm10"] + PROPERTY + ACCESS + AMENITY + SOCIO, False, None),
        ("M6", "+ year-quarter fixed effects", ["pm10"] + PROPERTY + ACCESS + AMENITY + SOCIO, True, None),
        ("M7", "+ borough fixed effects", ["pm10"] + PROPERTY + ACCESS + AMENITY + SOCIO, True, "oslaua"),
        ("M8", "LSOA instead of borough fixed effects (v1 model)",
         ["pm10"] + PROPERTY + ACCESS + AMENITY + SOCIO, True, "lsoacode"),
    ]
    models = []
    for code, label, xs, tfe, afe in steps:
        m = fe_ols(df, xs, tfe, afe)
        models.append(m)
        r = m["res"].loc["pm10"]
        print(f"{code} {label:<52} b_pm10 = {r.coef: .5f} (se {r.se:.5f})  R2 = {m['r2']:.3f}")
    b = [m["res"].at["pm10", "coef"] for m in models]
    se = [m["res"].at["pm10", "se"] for m in models]

    # --- sample choice: jobdensity and the 2015 cut ---
    X5 = ["pm10"] + PROPERTY + ACCESS + AMENITY + SOCIO
    X5_noj = [v for v in X5 if v != "jobdensity"]
    m5_noj = fe_ols(df, X5_noj)
    m5_full = fe_ols(full, X5_noj)
    m8_full = fe_ols(full, X5_noj, True, "lsoacode")

    # --- robustness ---
    m9 = fe_ols(df, X5 + ["nox"], True, "lsoacode")
    dc = add_crime(df)
    crime_models = None
    if dc is not None:
        crime_models = [fe_ols(dc, X5, True, "lsoacode"), fe_ols(dc, X5 + ["log_crime"], True, "lsoacode")]
    m8_la_cl = fe_ols(df, X5, True, "lsoacode", cluster="oslaua")
    m8_early = fe_ols(df[df["year"] <= 2009], X5, True, "lsoacode")
    m8_late = fe_ols(df[df["year"] >= 2010], X5, True, "lsoacode")
    df["la_year"] = df["oslaua"] + "_" + df["year"].astype(str)
    m8_lay = two_fe_ols(df, X5, "lsoacode", "la_year")

    # --- WTP from the v1 model ---
    mean_price = df["price"].mean()
    wtp = -b[7] * mean_price

    # --- write markdown ---
    var_rows = []
    for v, (desc, unit, src) in VARS.items():
        d = dc if v == "log_crime" and dc is not None else df
        if v not in d:
            continue
        model = ("Outcome" if v == "lnprice" else "M1" if v == "pm10" else "M9" if v == "nox"
                 else "M2" if v in PROPERTY else "M3" if v in ACCESS else "M4" if v in AMENITY
                 else "M5" if v in SOCIO else "Crime check")
        var_rows.append(f"| `{v}` | {desc} | {unit} | `{src}` | {model} | {d[v].mean():,.3f} | {d[v].std():,.3f} |")

    d01 = lambda i: f"{pct(b[i - 1])} to {pct(b[i])}"
    main_table = table(models, [s[0] for s in steps], ["pm10"] + PROPERTY + ACCESS + AMENITY + SOCIO)

    md = f"""# Model v1: Air Pollution and House Prices in London

Generated by `model_v1.py`. Outcome: log sale price. Coefficients ×100 give the % change in price
for a one-unit change in the variable.

## Sample

| Step | Sales |
|---|---|
| All sales in the pollution data | {info['n_raw']:,} |
| Prices trimmed (bottom and top 1%: below GBP {info['lo']:,.0f}, above GBP {info['hi']:,.0f}) | {info['n_trim']:,} |
| Only the 33 London boroughs (drops 3 local authorities outside London) | {info['n_london']:,} |
| 2005 to {LAST_YEAR}, no missing values | **{info['n_final']:,}** |

The final sample covers {info['n_la']} boroughs, {info['n_msoa']} MSOAs and {info['n_lsoa']:,} LSOAs.
City of London (about 1,000 sales) drops out because its unemployment rate is missing in all years.
This also removes its extreme job density (about 74, compared with 0.5 to 4 elsewhere).
Standard errors are clustered by MSOA, because PM10 comes from a small number of monitoring sites,
so nearby sales share the same pollution value.

### Why the sample ends in {LAST_YEAR}

`jobdensity` is missing for all sales in 2015. Keeping it means dropping 2015 (about 3% of sales).
We keep it because it matters for the PM10 estimate:

| M5 (no fixed effects) | Sample | PM10 effect |
|---|---|---|
| with `jobdensity` | 2005-{LAST_YEAR} | {pct(b[4])} |
| without `jobdensity` | 2005-{LAST_YEAR} | {pct(m5_noj['res'].at['pm10', 'coef'])} |
| without `jobdensity` | 2005-2015 | {pct(m5_full['res'].at['pm10', 'coef'])} |

`jobdensity` is significant (t = {models[4]['res'].at['jobdensity', 'coef'] / models[4]['res'].at['jobdensity', 'se']:.1f}),
and leaving it out changes the PM10 effect from {pct(b[4])} to {pct(m5_noj['res'].at['pm10', 'coef'])}.
Job density raises prices and, given the other controls, tends to be higher where PM10 is higher,
so without this control part of the job-density premium is picked up by PM10. In the final LSOA model the choice matters less:
with all of 2015 and without `jobdensity` the PM10 effect is {pct(m8_full['res'].at['pm10', 'coef'])},
compared with {pct(b[7])} in the v1 model.

## Variables

| Variable | Description | Unit | Built from | Enters in | Mean | Std. dev. |
|---|---|---|---|---|---|---|
{chr(10).join(var_rows)}

## Adding the controls step by step

PM10 effect = % change in price for 1 µg/m³ more PM10.

| Model | Added | PM10 effect | Std. error | R² |
|---|---|---|---|---|
""" + "\n".join(
        f"| {s[0]} | {s[1]} | {pct(b[i])}{stars(b[i], se[i])} | {se[i] * 100:.2f} | {models[i]['r2']:.3f} |"
        for i, s in enumerate(steps)) + f"""

**What each step changes:**

- **M1, PM10 only:** {pct(b[0])}. Without controls, more polluted areas look *more* expensive.
  This is not a causal effect: PM10 is highest near the centre, where homes cost more for other reasons.
- **M2, property characteristics** (type, new build, tenure): {d01(1)}. Central, polluted areas have more flats,
  which are cheaper than houses, so controlling for property type removes a discount and the PM10 effect rises.
- **M3, accessibility** (distance to centre, tube, bus): {d01(2)}. The sign turns negative. This is the key step:
  once we compare homes at the same distance from the centre and transport, more pollution goes with lower prices.
- **M4, environmental amenities** (park, Thames, heritage buildings): {d01(3)}. Small change, these amenities
  are not strongly related to PM10.
- **M5, socioeconomic controls** (borough pay, job density, unemployment, population; local education,
  minorities, lone parents): {d01(4)}. These remove differences in local demand and income.
- **M6, year-quarter fixed effects:** {d01(5)}. Prices rose strongly over time while PM10 fell, so without
  time effects the general price trend mixes with the pollution effect.
- **M7, borough fixed effects:** {d01(6)}. Now only sales within the same borough are compared. Permanent
  borough differences (schools, planning, reputation) no longer drive the result.
- **M8, LSOA fixed effects (v1 model):** {d01(7)}. Sales are compared within the same small neighbourhood,
  so PM10 is identified from changes over time within an LSOA. The local census shares hardly vary within
  an LSOA, so they are mostly absorbed and their coefficients should not be interpreted here.

## Full regression table

{main_table}

Standard errors clustered by MSOA in brackets. \\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01.
Base groups: flat (property type), leasehold (tenure). The constant is not shown.

## v1 model (M8)

ln(price) = β · PM10 + property + accessibility + amenities + socioeconomic controls
+ LSOA fixed effects + year-quarter fixed effects + error

- **PM10 effect: {pct(b[7])} per µg/m³** (standard error {se[7] * 100:.2f} percentage points).
- At the mean price of GBP {mean_price:,.0f}, a 1 µg/m³ reduction in PM10 is worth about
  **GBP {wtp:,.0f} per home**.
- R² = {models[7]['r2']:.3f}, {models[7]['n']:,} sales.

## Robustness checks

| Check | PM10 effect | Std. error | Sales |
|---|---|---|---|
| v1 model (M8) | {pct(b[7])} | {se[7] * 100:.2f} | {models[7]['n']:,} |
| M9: + NOx | {pct(m9['res'].at['pm10', 'coef'])} | {m9['res'].at['pm10', 'se'] * 100:.2f} | {m9['n']:,} |
| M9: NOx coefficient | {pct(m9['res'].at['nox', 'coef'])} | {m9['res'].at['nox', 'se'] * 100:.2f} | |
"""
    if crime_models:
        c0, c1 = crime_models
        md += (f"| Crime check: M8 on 2010-{LAST_YEAR}, without crime | {pct(c0['res'].at['pm10', 'coef'])} | "
               f"{c0['res'].at['pm10', 'se'] * 100:.2f} | {c0['n']:,} |\n"
               f"| Crime check: M8 on 2010-{LAST_YEAR}, with crime | {pct(c1['res'].at['pm10', 'coef'])} | "
               f"{c1['res'].at['pm10', 'se'] * 100:.2f} | {c1['n']:,} |\n"
               f"| Crime check: log crime coefficient | {c1['res'].at['log_crime', 'coef']:+.4f} | "
               f"{c1['res'].at['log_crime', 'se']:.4f} | |\n")
    def row(label, m):
        c, s_ = m["res"].at["pm10", "coef"], m["res"].at["pm10", "se"]
        return f"| {label} | {pct(c)}{stars(c, s_)} | {s_ * 100:.2f} | {m['n']:,} |\n"

    md += row("M8, standard errors clustered by borough (32 clusters)", m8_la_cl)
    md += row("M8, only 2005-2009", m8_early)
    md += row(f"M8, only 2010-{LAST_YEAR}", m8_late)
    md += row("M8 + borough-by-year fixed effects", m8_lay)
    md += f"""
\\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01.
Crime data only exist from 2010, so crime is tested on 2010-{LAST_YEAR} only. Compare the two crime rows
with each other (same sample), not with the full-sample v1 model.
Borough-by-year fixed effects remove any shock that hits a whole borough in a given year
(for example regeneration or large construction projects).
"""

    # --- verdict ---
    def coef(m):
        return m["res"].at["pm10", "coef"]

    def sig(m):
        t = abs(coef(m) / m["res"].at["pm10", "se"])
        return ("significant at 1%" if t > 2.576 else "significant at 5%" if t > 1.96
                else "significant at 10%" if t > 1.645 else "not significant")

    checks = [m9, m8_la_cl, m8_early, m8_late, m8_lay] + (list(crime_models) if crime_models else [])
    all_negative = all(coef(m) < 0 for m in checks)
    low = max(coef(m) for m in checks)            # smallest effect in absolute terms
    elast = -b[7] * df["pm10"].mean()

    md += f"""
## Verdict

**Direction: robust.** The PM10 effect is {'negative in every check' if all_negative else 'not negative in every check'}.
More pollution goes with lower prices, whichever controls, sample or standard errors we use.

**Unaffected by:**
- **NOx** ({pct(coef(m9))}, {sig(m9)}): the result is not driven by another pollutant.
- **Crime** ({pct(coef(crime_models[1])) if crime_models else 'n/a'} with crime vs {pct(coef(crime_models[0])) if crime_models else 'n/a'} without, on 2010-{LAST_YEAR}):
  crime does not explain the PM10 effect.
- **Clustering by borough** ({pct(coef(m8_la_cl))}, {sig(m8_la_cl)}): the significance does not
  depend on the choice of clusters.

**Size: uncertain.**
- Each period on its own gives a smaller effect ({pct(coef(m8_early))} for 2005-2009, {sig(m8_early)};
  {pct(coef(m8_late))} for 2010-{LAST_YEAR}, {sig(m8_late)}) than the pooled {pct(b[7])}.
- With borough-by-year fixed effects the effect falls to {pct(coef(m8_lay))} and is {sig(m8_lay)}.
  Either part of the M8 estimate comes from borough-wide shocks that move pollution and prices together,
  or these fixed effects absorb most of the PM10 variation (PM10 comes from about 60 to 74 monitoring
  sites, so much of it is borough-wide). The data cannot tell these two apart.

**Conclusion.** We keep M8 as the preferred model and read it as an upper estimate of the association:
1 µg/m³ more PM10 goes with roughly {abs(low) * 100:.2f}% to {abs(b[7]) * 100:.2f}% lower prices, about GBP {-low * mean_price:,.0f}
to GBP {wtp:,.0f} per home at the mean price. The implied elasticity at mean PM10 is {-elast:.3f}, smaller than the
-0.20 to -0.35 that Chay and Greenstone (2005) find with a quasi-experiment, and closer to the cross-sectional
estimates they criticise. Because PM10 is measured at the nearest monitoring site, measurement error likely
pushes our estimate towards zero. The estimate is a robust negative association, not a causal willingness
to pay; a clean causal estimate would need a quasi-experiment such as a policy that changed pollution in
some areas but not others.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE}")


if __name__ == "__main__":
    main()
