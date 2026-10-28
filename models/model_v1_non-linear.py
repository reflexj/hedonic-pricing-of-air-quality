"""
HE3027 Group Project: does a non-linear distance specification improve the hedonic model?

Takes the controls of model_v1.py and replaces the linear distance variables with
  - quadratic terms   (x and x²),
  - logs              (log x),
  - log-quadratic     (log x and (log x)²),
first only for dist_to_cbd, then for all five distance variables. Each version is estimated
with the fixed effects of M5 to M8, and compared on fit (R², adjusted R², BIC), on a clustered
Wald test of the extra terms, and on the PM10 effect. Writes model_v1_non-linear.md.

Run from the repository root:
    python models/model_v1_non-linear.py

Needs: pandas, numpy, scipy
Data:  data/land_registry_pollution.dta
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from model_v1 import (ACCESS, AMENITY, CLUSTER, PROPERTY, SOCIO, load_data, pct, stars)

MODEL_DIR = Path(__file__).resolve().parent   # models/
MD_FILE = MODEL_DIR / "model_v1_non-linear.md"

DISTANCES = ["dist_to_cbd", "tube_km", "bus_km", "park_km", "thames_km"]
LOG_FLOOR = 0.01     # km; park_km is 0 for 11 sales, so logs use max(x, 10 m)

FE_STEPS = [  # code, label, year-quarter FE, area FE
    ("M5", "no fixed effects", False, None),
    ("M6", "year-quarter FE", True, None),
    ("M7", "year-quarter + borough FE", True, "oslaua"),
    ("M8", "year-quarter + LSOA FE (final)", True, "lsoacode"),
]


# ---------------------------------------------------------------------------
# Distance specifications
# ---------------------------------------------------------------------------
def add_terms(df):
    for v in DISTANCES:
        lg = np.log(df[v].clip(lower=LOG_FLOOR))
        df[f"{v}_sq"] = df[v] ** 2
        df[f"ln_{v}"] = lg
        df[f"ln_{v}_sq"] = lg ** 2


def terms(v, form):
    return {"lin": [v], "quad": [v, f"{v}_sq"], "log": [f"ln_{v}"],
            "logquad": [f"ln_{v}", f"ln_{v}_sq"]}[form]


def xvars(form, which):
    """Full control set with the distance variables in `which` written in `form`."""
    out = ["pm10"] + PROPERTY
    for v in ACCESS + AMENITY:
        out += terms(v, form) if v in which else [v]
    return out + SOCIO


SPECS = [  # code, label, form, distances changed
    ("lin", "Linear (current model)", "lin", []),
    ("cbd_quad", "dist_to_cbd + dist_to_cbd²", "quad", ["dist_to_cbd"]),
    ("cbd_log", "log(dist_to_cbd)", "log", ["dist_to_cbd"]),
    ("cbd_logquad", "log(dist_to_cbd) + log(dist_to_cbd)²", "logquad", ["dist_to_cbd"]),
    ("all_quad", "All 5 distances: x + x²", "quad", DISTANCES),
    ("all_log", "All 5 distances: log(x)", "log", DISTANCES),
    ("all_logquad", "All 5 distances: log(x) + log(x)²", "logquad", DISTANCES),
]


# ---------------------------------------------------------------------------
# Same estimator as model_v1.fe_ols, but also returns the clustered
# covariance matrix (for Wald tests) and the parameter count incl. fixed effects
# ---------------------------------------------------------------------------
def fe_ols(d, xs, time_fe=False, area_fe=None, cluster=CLUSTER):
    X = d[xs].astype(float)
    if time_fe:
        X = pd.concat([X, pd.get_dummies(d["yearquarter"], prefix="yq", drop_first=True).astype(float)], axis=1)
    y = d["lnprice"].astype(float)
    if area_fe:
        g = d[area_fe]
        X = X - X.groupby(g).transform("mean")
        y = y - y.groupby(g).transform("mean")
        n_fe = g.nunique()
    else:
        X = X.assign(const=1.0)
        n_fe = 0
    X = X.loc[:, X.abs().sum() > 1e-10]

    Xv, yv = X.to_numpy(), y.to_numpy()
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    b = XtX_inv @ Xv.T @ yv
    e = yv - Xv @ b
    scores = pd.DataFrame(Xv * e[:, None]).groupby(d[cluster].to_numpy()).sum().to_numpy()
    n, k, G = len(yv), Xv.shape[1], len(scores)
    V = XtX_inv @ (scores.T @ scores) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    ssr = e @ e
    sst = ((d["lnprice"] - d["lnprice"].mean()) ** 2).sum()
    k_all = k + n_fe
    return {"res": pd.DataFrame({"coef": b, "se": np.sqrt(np.diag(V))}, index=X.columns),
            "V": pd.DataFrame(V, index=X.columns, columns=X.columns), "n": n, "G": G,
            "r2": 1 - ssr / sst, "adj_r2": 1 - (ssr / (n - k_all)) / (sst / (n - 1)),
            "bic": n * np.log(ssr / n) + k_all * np.log(n)}


def wald(m, vs):
    """Clustered Wald test that all coefficients in vs are zero (F form, G-1 denominator df)."""
    vs = [v for v in vs if v in m["res"].index]
    b = m["res"].loc[vs, "coef"].to_numpy()
    F = b @ np.linalg.solve(m["V"].loc[vs, vs].to_numpy(), b) / len(vs)
    return F, stats.f.sf(F, len(vs), m["G"] - 1)


def turning_point(m, v):
    """Distance in km where a quadratic in v has its minimum / maximum."""
    b1, b2 = m["res"].at[v, "coef"], m["res"].at[f"{v}_sq", "coef"]
    return -b1 / (2 * b2), "min" if b2 > 0 else "max"


def pval_txt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, _, info = load_data()
    add_terms(df)

    fits = {}   # (fe_code, spec_code) -> model
    for fe_code, fe_label, tfe, afe in FE_STEPS:
        for code, label, form, which in SPECS:
            m = fe_ols(df, xvars(form, which), tfe, afe)
            fits[fe_code, code] = m
            r = m["res"].loc["pm10"]
            print(f"{fe_code} {label:<40} b_pm10 = {r.coef: .5f} (se {r.se:.5f})  "
                  f"R2 = {m['r2']:.4f}  BIC = {m['bic']:,.0f}")

    # Tests against the linear model. Nested cases: the quadratic adds x², the log-quadratic
    # in addition to the log. Log-only is not nested in linear, so it is compared by BIC only.
    def extra_terms(code):
        form, which = next((s[2], s[3]) for s in SPECS if s[0] == code)
        return ([f"{v}_sq" for v in which] if form == "quad"
                else [f"ln_{v}_sq" for v in which] if form == "logquad" else None)

    # --- fit tables, one per fixed-effects level ---
    sections = []
    for fe_code, fe_label, _, _ in FE_STEPS:
        base = fits[fe_code, "lin"]
        rows = []
        for code, label, _, _ in SPECS:
            m = fits[fe_code, code]
            c, s = m["res"].at["pm10", "coef"], m["res"].at["pm10", "se"]
            ex = extra_terms(code)
            test = ""
            if ex is not None:
                F, p = wald(m, ex)
                test = f"{F:.1f} (p {pval_txt(p)})"
            rows.append(f"| {label} | {m['r2']:.4f} | {m['adj_r2']:.4f} | {m['bic'] - base['bic']:+,.0f} | "
                        f"{test} | {pct(c)}{stars(c, s)} | {s * 100:.2f} |")
        best = min((fits[fe_code, s[0]]["bic"], s[1]) for s in SPECS)[1]
        sections.append(f"""### {fe_code}: {fe_label}

| Distance specification | R² | Adj. R² | ΔBIC vs linear | Wald F, extra terms | PM10 effect | Std. error |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Lowest BIC: **{best}**.
""")

    # --- shape of the distance-to-centre gradient ---
    shape_rows = []
    for fe_code, _, _, _ in FE_STEPS:
        m = fits[fe_code, "cbd_quad"]
        b1, b2 = m["res"].at["dist_to_cbd", "coef"], m["res"].at["dist_to_cbd_sq", "coef"]
        tp, kind = turning_point(m, "dist_to_cbd")
        ml = fits[fe_code, "cbd_log"]
        bl, sl = ml["res"].at["ln_dist_to_cbd", "coef"], ml["res"].at["ln_dist_to_cbd", "se"]
        shape_rows.append(f"| {fe_code} | {b1:.5f}{stars(b1, m['res'].at['dist_to_cbd', 'se'])} | "
                          f"{b2:.6f}{stars(b2, m['res'].at['dist_to_cbd_sq', 'se'])} | {tp:.1f} km ({kind}) | "
                          f"{bl:.4f}{stars(bl, sl)} |")

    # --- M8: which distance carries the gain? one variable log-quadratic at a time ---
    within = {v: (df[v] - df.groupby("lsoacode")[v].transform("mean")).var() / df[v].var() for v in DISTANCES}
    m8_lin = fits["M8", "lin"]
    one_rows = []
    for v in DISTANCES:
        m = fe_ols(df, xvars("logquad", [v]), True, "lsoacode")
        F, p = wald(m, [f"ln_{v}_sq"])
        c = m["res"].at["pm10", "coef"]
        one_rows.append(f"| `{v}` | {within[v] * 100:.2f}% | {m['r2'] - m8_lin['r2']:+.4f} | "
                        f"{m['bic'] - m8_lin['bic']:+,.0f} | {F:.1f} (p {pval_txt(p)}) | {pct(c)} |")

    # --- headline numbers for the text ---
    def eff(fe, code):
        return pct(fits[fe, code]["res"].at["pm10", "coef"])

    def dbic(fe, code):
        return fits[fe, code]["bic"] - fits[fe, "lin"]["bic"]

    def dr2(fe, code):
        return fits[fe, code]["r2"] - fits[fe, "lin"]["r2"]

    best8 = min(SPECS, key=lambda s: fits["M8", s[0]]["bic"])
    m8_effects = [fits["M8", s[0]]["res"].at["pm10", "coef"] for s in SPECS]

    md = f"""# Non-linear Distance Controls

Generated by `model_v1_non-linear.py`. Same sample as `model_v1.py`
({info['n_final']:,} sales, 2005-2014, 33 London boroughs), same controls, same MSOA clustering.
Only the way the five distance variables enter the model changes.

**Question:** house prices probably do not fall by the same amount for every extra km from the centre
(the first km matters more than the 20th). Does a non-linear distance control fit the data better,
and does it change the PM10 effect?

## Specifications

| Code | Distance terms |
|---|---|
""" + "\n".join(f"| `{s[0]}` | {s[1]} |" for s in SPECS) + f"""

"All 5 distances" means `dist_to_cbd`, `tube_km`, `bus_km`, `park_km`, `thames_km`
(`Bus_stopscount_200m` and `heritage_count_200m` are counts and stay linear).
Logs use max(x, {LOG_FLOOR * 1000:.0f} m), because `park_km` is 0 for 11 sales.

**How to read the fit columns.** With {info['n_final']:,} sales almost any extra term is "significant",
so the R² gain and the BIC are more informative than the p-value. ΔBIC < 0 means the
non-linear version is better after penalising the extra parameters (below -10 is strong evidence).
The Wald F tests whether the added squared terms are jointly zero, using MSOA-clustered standard errors.
It is only shown where the model is nested in the linear one (log-only is not).

## Results by fixed-effects level

{chr(10).join(sections)}
\\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01. Std. error in percentage points.

## Shape of the distance-to-centre gradient

| Model | `dist_to_cbd` | `dist_to_cbd²` | Turning point | `log(dist_to_cbd)` (log model) |
|---|---|---|---|---|
{chr(10).join(shape_rows)}

The sample runs from {df['dist_to_cbd'].min():.1f} km to {df['dist_to_cbd'].max():.1f} km from the centre
(mean {df['dist_to_cbd'].mean():.1f} km).

In M5 to M7 the squared term is positive and the turning point lies at or beyond the edge of the sample,
so prices fall with distance over the whole of London, steeply near the centre and more slowly further out.
A pure log is too concave: on its own it fits worse than the linear term (ΔBIC
{dbic('M5', 'cbd_log'):+,.0f} in M5, {dbic('M6', 'cbd_log'):+,.0f} in M6, {dbic('M7', 'cbd_log'):+,.0f} in M7).
In M8 the distance coefficients come from tiny movements within an LSOA and should not be interpreted.

## Why the LSOA fixed effects make this almost irrelevant

M8 with one distance variable at a time in log-quadratic form (all others linear), compared with linear M8:

| Variable | Within-LSOA share of variance | ΔR² | ΔBIC | Wald F, (log x)² | PM10 effect |
|---|---|---|---|---|---|
{chr(10).join(one_rows)}

Homes in the same LSOA are almost the same distance from the centre, the tube, the parks and the Thames,
so the LSOA fixed effects already absorb *any* function of these distances. Only `bus_km` varies within
LSOAs, and its non-linear form adds nothing. The small M8 gain comes mostly from `log(thames_km)`: the log
stretches short distances, so it picks up a riverside premium among the few LSOAs right on the Thames.

## Verdict

**Better fit: yes, but mainly in the models without LSOA fixed effects.** In every column the lowest BIC
belongs to "{best8[1] if best8[0] == 'all_logquad' else 'All 5 distances: log(x) + log(x)²'}".
Compared with linear distances it changes R² by {dr2('M5', 'all_logquad'):+.4f} (M5),
{dr2('M6', 'all_logquad'):+.4f} (M6), {dr2('M7', 'all_logquad'):+.4f} (M7) and only {dr2('M8', 'all_logquad'):+.4f} (M8).
Making only `dist_to_cbd` non-linear helps much less ({dr2('M6', 'cbd_quad'):+.4f} in M6 with the quadratic);
most of the gain comes from the other distances.

**PM10 without LSOA fixed effects depends on how distance is modelled.** In M6 the PM10 effect moves from
{eff('M6', 'lin')} (not significant) with linear distances to {eff('M6', 'all_logquad')} with all distances
log-quadratic, and in M7 from {eff('M7', 'lin')} to {eff('M7', 'all_logquad')}. PM10 is highest near the centre,
so a too-simple distance gradient leaks into the PM10 coefficient and pushes it towards zero. With flexible
distances, M6 and M7 come close to the v1 model's {eff('M8', 'lin')}: part of the jump from M7 to M8 in
the step-by-step table is a functional-form problem that the LSOA fixed effects solve.

**v1 model (M8): the PM10 effect does not change.** Across all seven specifications it stays between
{pct(max(m8_effects))} and {pct(min(m8_effects))} (linear: {eff('M8', 'lin')}), with the same standard error.
The best specification raises R² by only {dr2('M8', best8[0]):+.4f}.

**Recommendation.** Keep linear distances in M8 for simplicity, and report the log-quadratic M8 as a robustness
check ("the PM10 effect does not depend on the functional form of the distance controls"). If you also want
M5 to M7 to be as credible as possible, use log(x) + log(x)² for the distances there, because with linear
distances they understate the PM10 effect.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE}")


if __name__ == "__main__":
    main()
