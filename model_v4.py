"""
HE3027 Group Project: model v4, green space as an instrument for PM10 (2SLS).

Idea (suggested by the lecturer): green space near a home lowers PM10, and green space affects house prices ONLY
through PM10 (exclusion restriction). Then green space can be used as an instrument for PM10, which removes the bias
from PM10 being measured with error at the nearest monitoring site and from unobserved local factors.

The model is the v3 model (3-year average PM10, v3 controls, MSOA-clustered standard errors) without the green terms
and without the interactions, because green space is now the instrument and cannot also be a control. Three settings:

  A. v3 setting: LSOA and year-quarter fixed effects, instrument = green share within 500 m.
     PM10 does not vary within an LSOA in a given year (it is the value of the nearest monitoring site) and green
     space does not change over time, so the instrument can only explain PM10 by chance. Reported to show this.
  B. Between areas: borough instead of LSOA fixed effects, instrument = green share within 500 m. This uses the
     differences in green space and PM10 between neighbourhoods of the same borough.
  C. Green space × year: LSOA fixed effects, the green share as a control, instruments = green share × year
     dummies. Asks whether greener neighbourhoods saw different PM10 changes, and allows green space to raise prices
     directly as long as that premium is constant over time (weaker exclusion restriction than A and B).

For every setting: first stage, its F statistic (clustered), reduced form, 2SLS, the OLS with the same fixed
effects, an Anderson-Rubin confidence interval (valid even with a weak instrument) and, in C, a Hansen J test.
Two checks of the exclusion restriction: does green space still move prices when PM10 is held fixed, and does
green space predict the other price determinants (balance test)?

Writes model_v4.md and figures/v4_green_iv.png.

Run from the Code folder:
    python model_v4.py

Needs: pandas, numpy, matplotlib, scipy, shapely, pyproj, pyshp
Data:  as model_v3.py (data/road_features.csv and data/green_features.csv are reused)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from model_v1 import CLUSTER, CODE_DIR, fe_ols, pct
from model_v1_income_groups import gbp, p_fmt
from model_v1_pm10_smoothing import fe_iv
from model_v2 import coef, sig_txt, with_pm
from model_v3 import LABELS, V3_CONTROLS, finish, prepare

MD_FILE = CODE_DIR / "model_v4.md"
FIG_DIR = CODE_DIR / "figures"
FIG_FILE = FIG_DIR / "v4_green_iv.png"

STRONG_F = 10            # rule of thumb for a strong instrument (Staiger and Stock 1997)
AR_GRID = np.linspace(-0.20, 0.20, 4001)   # PM10 coefficients tried for the Anderson-Rubin interval (-20% to +20%)
X_MAX = 12               # x-axis of the figure: -12% to +12% per µg/m³


# ---------------------------------------------------------------------------
# IV estimation: fixed effects and controls are partialled out first (Frisch-Waugh-Lovell), then 2SLS
# ---------------------------------------------------------------------------
def partial_out(d, cols, controls, area_fe):
    """Remove the area means, then the year-quarter dummies and the controls, from every column in `cols`."""
    g = d[area_fe]
    W = pd.concat([d[controls].astype(float),
                   pd.get_dummies(d["yearquarter"], prefix="yq", drop_first=True).astype(float)], axis=1)
    W = W - W.groupby(g).transform("mean")
    W = W.loc[:, W.abs().sum() > 1e-10].to_numpy()
    out = {}
    for c in cols:
        x = d[c].astype(float)
        x = (x - x.groupby(g).transform("mean")).to_numpy()
        out[c] = x - W @ np.linalg.lstsq(W, x, rcond=None)[0]
    return out, W.shape[1]


def cl_sum(a, cl):
    """Sum the rows of `a` (n x k) within clusters."""
    return pd.DataFrame(a).groupby(cl).sum().to_numpy()


def small_sample(n, k, G):
    return (G / (G - 1)) * ((n - 1) / (n - k))


def ols_part(y, Z, cl, k_w):
    """OLS of y on Z (both already partialled), clustered covariance."""
    ZZ_inv = np.linalg.inv(Z.T @ Z)
    b = ZZ_inv @ Z.T @ y
    e = y - Z @ b
    S = cl_sum(Z * e[:, None], cl)
    V = ZZ_inv @ (S.T @ S) @ ZZ_inv * small_sample(len(y), Z.shape[1] + k_w, len(S))
    return b, V


def wald(b, V):
    W = float(b @ np.linalg.solve(V, b))
    return W, stats.chi2.sf(W, len(b))


def iv(d, instruments, controls, area_fe="lsoacode", cluster=CLUSTER, endog="pm10"):
    """2SLS of lnprice on `endog`, instrumented by `instruments`, with `controls`, year-quarter dummies and
    `area_fe` absorbed. Returns the 2SLS estimate, the first stage, the reduced form, the Anderson-Rubin interval
    and (with more than one instrument) the Hansen J test."""
    P, k_w = partial_out(d, ["lnprice", endog] + instruments, controls, area_fe)
    y, x = P["lnprice"], P[endog]
    Z = np.column_stack([P[z] for z in instruments])
    cl = pd.factorize(d[cluster])[0]
    n, q, G = len(y), len(instruments), cl.max() + 1

    # first stage: PM10 on the instruments
    pi, V_pi = ols_part(x, Z, cl, k_w)
    W_fs, p_fs = wald(pi, V_pi)
    # reduced form: price on the instruments
    rho, V_rho = ols_part(y, Z, cl, k_w)
    # 2SLS
    xhat = Z @ pi
    b = float(xhat @ y / (xhat @ x))
    e = y - b * x
    s = cl_sum((xhat * e)[:, None], cl)
    se = float(np.sqrt((s.T @ s)[0, 0] / (xhat @ x) ** 2 * small_sample(n, 1 + k_w, G)))

    # Anderson-Rubin: the PM10 coefficients b0 for which the instruments do not explain lnprice - b0 * PM10
    # (regression of y - b0*x on Z). Coefficients and cluster sums are linear in b0, so each grid point is cheap.
    ZZ_inv = np.linalg.inv(Z.T @ Z)
    Sy = cl_sum(Z * (y - Z @ rho)[:, None], cl)
    Sx = cl_sum(Z * (x - Z @ pi)[:, None], cl)
    c = small_sample(n, q + k_w, G)
    ar_keep = []
    for b0 in AR_GRID:
        g0 = rho - b0 * pi
        S = Sy - b0 * Sx
        ar_keep.append(wald(g0, ZZ_inv @ (S.T @ S) @ ZZ_inv * c)[1] > 0.05)
    ar = ar_text(np.array(ar_keep))

    # Hansen J (over-identification), from the 2SLS residuals
    J = pJ = None
    if q > 1:
        S = cl_sum(Z * e[:, None], cl)
        gbar = S.sum(axis=0)
        J = float(gbar @ np.linalg.solve(S.T @ S, gbar))
        pJ = stats.chi2.sf(J, q - 1)

    return {"coef": b, "se": se, "n": n, "q": q, "pi": pi, "pi_se": np.sqrt(np.diag(V_pi)), "F": W_fs / q,
            "p_fs": p_fs, "rho": rho, "rho_se": np.sqrt(np.diag(V_rho)), "p_rf": wald(rho, V_rho)[1],
            "ar": ar, "J": J, "pJ": pJ}


def ar_text(keep):
    """The Anderson-Rubin confidence set from the grid of values that are not rejected."""
    if not keep.any():
        return "empty (every value rejected)"
    if keep[0] and keep[-1]:
        return "unbounded"
    lo, hi = AR_GRID[keep].min(), AR_GRID[keep].max()
    if keep[0]:
        return f"below {pct(hi)} (unbounded)"
    if keep[-1]:
        return f"above {pct(lo)} (unbounded)"
    inside = keep[np.argmax(keep):len(keep) - np.argmax(keep[::-1])]
    return f"{pct(lo)} to {pct(hi)}" + ("" if inside.all() else " (with gaps)")


def ols(d, controls, area_fe="lsoacode", cluster=CLUSTER, extra=()):
    return fe_ols(d, ["pm10", *extra] + controls, time_fe=True, area_fe=area_fe, cluster=cluster)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(rows):
    """rows: (label, coef, se, kind) with kind 'ols' or 'iv'; IV rows with a weak first stage are hollow."""
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#52514e", linewidth=0.8)
    for yi, (lab, c, s, kind, strong) in zip(y, rows):
        col = "#2a78d6" if kind == "ols" else "#eb6834"
        if abs(c * 100) > X_MAX:   # estimate off the scale: arrow at the edge, value as text
            edge = np.sign(c) * X_MAX
            ax.annotate("", (edge, yi), xytext=(edge - np.sign(c) * 2.5, yi),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
            ax.annotate(f"{c * 100:+.1f}% (s.e. {s * 100:.1f}), off scale", (edge - np.sign(c) * 2.7, yi),
                        ha="right" if c > 0 else "left", va="center", fontsize=8, color="#0b0b0b")
            continue
        ax.errorbar(c * 100, yi, xerr=1.96 * s * 100, fmt="o", color=col, ecolor="#0b0b0b", elinewidth=1,
                    capsize=3, markersize=7, zorder=3, markerfacecolor=col if strong else "white")
        ax.annotate(f"{c * 100:+.2f}%", (c * 100, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color="#0b0b0b")
    ax.set_xlim(-X_MAX, X_MAX)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("% change in price per +1 µg/m³ PM10 (95% CI)")
    ax.set_title("PM10 effect: OLS and 2SLS with green space as the instrument", loc="left", fontsize=11,
                 color="#0b0b0b")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, f"Blue: OLS. Orange: 2SLS (hollow: weak instrument, first-stage F < {STRONG_F}). "
             "Standard errors clustered by MSOA.\nWith a weak instrument these intervals are not reliable; "
             "see the Anderson-Rubin intervals in model_v4.md.", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = prepare()
    d = finish(with_pm(df.dropna(subset=["green_500", "dist_major_m"]), "pm10_ma3"))      # v3 sample
    years = sorted(d["year"].unique())
    for yr in years[1:]:
        d[f"green_y{yr}"] = d["green10"] * (d["year"] == yr)
    GxY = [f"green_y{yr}" for yr in years[1:]]
    mean_price = d["price"].mean()
    C = V3_CONTROLS

    def within(v, fe):
        x = d[v].astype(float)
        return (x - x.groupby(d[fe]).transform("mean")).var() / x.var()
    var_left = {(v, fe): within(v, fe) for v in ["pm10", "green10"] for fe in ["lsoacode", "msoa11", "oslaua",
                                                                                "lsoa_year"]}

    # ---------------- OLS benchmarks (same controls, no green terms) ----------------
    ols_l = ols(d, C)                                       # LSOA FE
    ols_b = ols(d, C, area_fe="oslaua")                     # borough FE
    ols_l_g = ols(d, C, extra=["green10"])                  # LSOA FE, green share as a control
    ols_b_g = ols(d, C, area_fe="oslaua", extra=["green10"])
    print("OLS done")

    # ---------------- 2SLS ----------------
    A = iv(d, ["green10"], C)                                           # v3 setting
    B = iv(d, ["green10"], C, area_fe="oslaua")                         # between areas
    B_la = iv(d, ["green10"], C, area_fe="oslaua", cluster="oslaua")    # same, borough clusters
    B_ms = iv(d, ["green10"], C, area_fe="msoa11")                      # MSOA FE (in between A and B)
    Cg = iv(d, GxY, C + ["green10"])                                    # green × year
    print("IV done")
    # check against the 2SLS function of model_v1_pm10_smoothing.py (one instrument, LSOA FE)
    chk = fe_iv(d, "pm10", "green10", C)
    assert abs(chk["coef"] - A["coef"]) < 1e-6 * max(1, abs(A["coef"])), (chk["coef"], A["coef"])

    # ---------------- exclusion-restriction checks ----------------
    # 1) does green space still move prices with PM10 held fixed?
    g_l, g_b = coef(ols_l_g, "green10"), coef(ols_b_g, "green10")
    # 2) balance: does green space predict the price implied by the other controls?
    idx_fit = fe_ols(d, C, time_fe=True, area_fe=None)
    Xc = d[C].astype(float).assign(const=1.0)
    d["price_index"] = Xc[idx_fit["res"].index.intersection(Xc.columns)].to_numpy() @ \
        idx_fit["res"].loc[idx_fit["res"].index.intersection(Xc.columns), "coef"].to_numpy()
    bal = {}
    for fe in ["oslaua", "lsoacode"]:
        m = fe_ols(d.assign(lnprice=d["price_index"]), ["green10"], time_fe=True, area_fe=fe)
        bal[fe] = coef(m, "green10")
    bal_vars = {}
    for v, lab in [("ln_dist_to_cbd", "log distance to centre"), ("percent_noedu", "share without qualifications"),
                   ("road_0_50", "within 50 m of a main road"), ("house", "house (not a flat)")]:
        m = fe_ols(d.assign(lnprice=d[v].astype(float)), ["green10"], time_fe=True, area_fe="oslaua")
        bal_vars[lab] = (coef(m, "green10"), d[v].astype(float).std())
    print("checks done")

    plot([("OLS, LSOA FE (v3 without green terms)", *coef(ols_l, "pm10"), "ols", True),
          ("A. 2SLS, LSOA FE", A["coef"], A["se"], "iv", A["F"] >= STRONG_F),
          ("OLS, borough FE", *coef(ols_b, "pm10"), "ols", True),
          ("B. 2SLS, borough FE", B["coef"], B["se"], "iv", B["F"] >= STRONG_F),
          ("   B with MSOA FE", B_ms["coef"], B_ms["se"], "iv", B_ms["F"] >= STRONG_F),
          ("C. 2SLS, green × year, LSOA FE", Cg["coef"], Cg["se"], "iv", Cg["F"] >= STRONG_F)])

    # ===================== Markdown =====================
    def strength(r):
        return "strong" if r["F"] >= STRONG_F else "weak"

    def iv_cell(r):
        return f"{r['coef'] * 100:+.3f} ({r['se'] * 100:.3f})"

    def ols_cell(m):
        c, s = coef(m, "pm10")
        return f"{c * 100:+.3f} ({s * 100:.3f})"

    def fs_cell(r):
        if r["q"] == 1:
            return f"{r['pi'][0]:+.4f} ({r['pi_se'][0]:.4f})"
        return f"{r['q']} coefficients, joint p {p_fmt(r['p_fs'])}"

    def rf_cell(r):
        if r["q"] == 1:
            return f"{r['rho'][0] * 100:+.3f} ({r['rho_se'][0] * 100:.3f})"
        return f"joint p {p_fmt(r['p_rf'])}"

    settings = [
        ("A. v3 setting: LSOA FE, green share", A, ols_l),
        ("B. Between areas: borough FE, green share", B, ols_b),
        ("B with MSOA FE", B_ms, None),
        ("B with borough clusters", B_la, None),
        ("C. Green × year: LSOA FE, green share as a control", Cg, ols_l_g),
    ]
    main_rows = []
    for lab, r, m in settings:
        main_rows.append(f"| {lab} | {fs_cell(r)} | {r['F']:.1f} ({strength(r)}) | {rf_cell(r)} | {iv_cell(r)} | "
                         f"{r['ar']} | {ols_cell(m) if m is not None else ''} |")
    cy_rows = [f"| {yr} | {Cg['pi'][i]:+.4f} ({Cg['pi_se'][i]:.4f}) |" for i, yr in enumerate(years[1:])]
    bal_rows = [f"| {lab} | {c:+.4f} ({s:.4f}) | {sig_txt(c, s)} | {abs(c) / sd * 100:.1f}% of a s.d. |"
                for lab, ((c, s), sd) in bal_vars.items()]

    n_d = len(d)
    pm_mean = d['pm10'].mean()
    IVS = [A, B, B_ms, Cg]
    max_F = max(r['F'] for r in IVS)
    if max_F >= STRONG_F:
        print('WARNING: an instrument is strong now; the verdict text assumes weak instruments')
    md = f"""# Model v4: Green Space as an Instrument for PM10

Generated by `model_v4.py`. Outcome: log sale price. Coefficients ×100 give the % change in price per +1 µg/m³
PM10. Sample: the v3 sample ({n_d:,} sales, {years[0]}-{years[-1]}, 3-year average PM10). Standard errors
clustered by MSOA unless stated otherwise.

## 1. The idea

Our OLS estimates (v1 to v3) can be biased for two reasons:

- **Measurement error.** PM10 is the value of the nearest of about 60 to 74 monitoring sites, not of the home. This
  biases the effect towards zero (attenuation).
- **Omitted variables.** Anything that changes prices and PM10 at the same time and is not in our controls.

An **instrument** is a variable that (1) moves PM10 (**relevance**) and (2) affects prices **only** through PM10
(**exclusion restriction**). Then 2SLS uses only the part of PM10 that the instrument explains:

- **First stage:** PM10 = π · green + controls + fixed effects
- **Second stage:** ln(price) = β · predicted PM10 + controls + fixed effects

The lecturer's suggestion: **green space** (share of land within 500 m of the postcode that is green, OS Open
Greenspace, the same measure as in v3) lowers PM10 because vegetation filters particles, and it raises prices only
because of cleaner air. Equivalent check: the reduced form (price on green) should equal β × π.

The exclusion restriction is a strong assumption. Green space is also an amenity in itself (parks, views,
recreation, quiet), so it can raise prices directly. Section 4 tests this as far as the data allow.

**Model.** The v3 model ({len(C)} controls: property type, distances, bus access, census shares, borough economy,
main roads, road noise) with the 3-year average of PM10. The green share and the low-green dummy are **not** in the
model (green space is the instrument), and the interactions PM10 × house and PM10 × low green are left out, so the
2SLS estimates one average PM10 effect. The OLS benchmark uses exactly the same controls.

## 2. Why the v3 setting cannot identify the IV

In v3, PM10 is identified from **changes over time within an LSOA**: PM10 is the same for every sale in an LSOA in a
given year (share of PM10 variance left within LSOA-years: {var_left[('pm10', 'lsoa_year')] * 100:.1f}%), and LSOA and year-quarter fixed effects remove the
level differences between LSOAs and the London-wide trend. Green space does **not** change over time (one current
OS release). After the LSOA fixed effects, green space only differs between postcodes of the same LSOA, and these
postcodes all get the same PM10. So in setting A, green space can explain PM10 only through which years the sales in
an LSOA happen to fall in. The first stage is expected to be zero, and it is
(F = {A['F']:.2f}).

Two settings that can work:

- **B. Between areas (borough fixed effects).** Compares neighbourhoods of the same borough with more and less
  green space. Within boroughs, {var_left[('pm10', 'oslaua')] * 100:.0f}% of the PM10 variance and
  {var_left[('green10', 'oslaua')] * 100:.0f}% of the green variance are left (within LSOAs: {var_left[('pm10', 'lsoacode')] * 100:.0f}% and
  {var_left[('green10', 'lsoacode')] * 100:.0f}%), so there is variation to use. But this gives up the LSOA fixed
  effects, which the earlier versions showed to be essential: without them the PM10 effect depends heavily on the
  location controls (`model_v1.md`).
- **C. Green space × year.** Keeps the LSOA fixed effects and asks whether greener places saw different PM10
  **changes** over 2007-2014. The green share stays in the model as a control, so green space may raise prices
  directly; only its effect on price **changes** is excluded. This is a weaker (more believable) exclusion
  restriction, but it needs green space to change the PM10 **trend**, which the vegetation story does not predict.

## 3. Results

| Setting | First stage: green → PM10 (µg/m³ per +10 pp green) | First-stage F | Reduced form: green → price (% per +10 pp) | **2SLS PM10 effect** | Anderson-Rubin 95% interval | OLS, same FE |
|---|---|---|---|---|---|---|
{chr(10).join(main_rows)}

PM10 effects in % per µg/m³ (standard error); first stage in µg/m³ PM10 per +10 percentage points of green
space. F: clustered Wald F statistic of the excluded instrument(s); below {STRONG_F} the instrument is weak and the
2SLS estimate and its standard error are not reliable. The Anderson-Rubin interval is valid even with a weak
instrument: it contains every PM10 effect that the data cannot reject at 5%. Values between
{pct(AR_GRID[0])} and {pct(AR_GRID[-1])} were tried.

![OLS vs 2SLS](figures/v4_green_iv.png)

**Setting C, first stage by year** (PM10 change in µg/m³ per +10 pp green space, relative to {years[0]}):

| Year | Green × year |
|---|---|
{chr(10).join(cy_rows)}

Hansen J test of over-identification (do the {Cg['q']} instruments agree?): J = {Cg['J']:.2f}, df {Cg['q'] - 1},
p {p_fmt(Cg['pJ'])}.

## 4. Is the exclusion restriction plausible?

**Check 1: does green space still move prices when PM10 is held fixed?** If green space affected prices only
through PM10, its coefficient should be zero once PM10 is in the regression.

| Fixed effects | Green share (per +10 pp), PM10 held fixed | |
|---|---|---|
| Borough | {pct(g_b[0])} (s.e. {g_b[1] * 100:.3f} pp) | {sig_txt(*g_b)} |
| LSOA | {pct(g_l[0])} (s.e. {g_l[1] * 100:.3f} pp) | {sig_txt(*g_l)} |

**Check 2: is green space as good as random?** An instrument should not be related to the other determinants of
prices. The "price index" is the price predicted by the {len(C)} controls alone (OLS without PM10 and without fixed
effects). If green space predicts it, greener and less green places differ in other ways too.

| Fixed effects | Price index on green share (% per +10 pp) | |
|---|---|---|
| Borough | {pct(bal['oslaua'][0])} (s.e. {bal['oslaua'][1] * 100:.3f} pp) | {sig_txt(*bal['oslaua'])} |
| LSOA | {pct(bal['lsoacode'][0])} (s.e. {bal['lsoacode'][1] * 100:.3f} pp) | {sig_txt(*bal['lsoacode'])} |

Single controls on the green share (borough FE, per +10 pp green):

| Control | Coefficient (s.e.) | | Size |
|---|---|---|---|
{chr(10).join(bal_rows)}

## 5. Verdict

- **Relevance fails: green space is a weak instrument in every setting.** No setting reaches a first-stage F of
  {STRONG_F} (highest: {max_F:.1f}, setting B). With LSOA fixed effects (A) this is by construction (section 2).
  Between neighbourhoods of the same borough (B), greener places do have slightly lower PM10, with the expected sign
  ({B['pi'][0]:+.3f} µg/m³ per +10 pp green space, {abs(B['pi'][0]) / pm_mean * 100:.1f}% of mean PM10), but this is
  far too little to predict PM10 reliably. Green space also did not change the PM10 **trend** (C: joint p
  {p_fmt(Cg['p_fs'])}).
- **So the 2SLS estimates say nothing about the PM10 effect.** They jump between {pct(min(r['coef'] for r in IVS))} and
  {pct(max(r['coef'] for r in IVS))} per µg/m³ with huge standard errors. The Anderson-Rubin intervals, which stay valid
  with a weak instrument, are unbounded (A, C) or run from {B['ar']} (B): they contain both zero and our OLS estimate
  ({pct(coef(ols_l, 'pm10')[0])}). The IV can neither confirm nor reject the v3 result.
- **The exclusion restriction cannot be confirmed either.** With PM10 held fixed, green space has no significant
  direct price effect (check 1), but the estimates are imprecise. Greener postcodes differ in other ways: they are
  further from the centre, have more houses and more residents without qualifications, and within LSOAs the price
  predicted by the controls rises with green space ({pct(bal['lsoacode'][0])} per +10 pp, {sig_txt(*bal['lsoacode'])}).
  If they differ in observed price determinants, they probably differ in unobserved ones too.
- **Why it does not work here.** Our PM10 is measured at monitoring sites, often kilometres away, and green space
  is measured within 500 m and does not change over time. The mechanism (vegetation filtering particles locally)
  operates at a scale our PM10 measure cannot see.
- **Consequence for the report.** v3 stays the main model. v4 is a documented robustness attempt: the
  lecturer's instrument was tested and is too weak with our data, so it cannot be used to give the PM10 effect a
  causal interpretation. A working IV would need PM10 modelled at the address (e.g. LAEI) or a source of PM10 change
  that differs between areas and is unrelated to other price drivers.

## Limitations

- **Scale.** Green space is measured within 500 m of the postcode, PM10 at the nearest monitoring site, often
  several kilometres away. Vegetation near a home cannot change the reading of a distant monitor, so even if green
  space does clean the air locally, our PM10 measure cannot show it. A modelled PM10 surface (e.g. the London
  Atmospheric Emissions Inventory, LAEI, at 20 m resolution) would be needed for a real test of the mechanism.
- **Green space is fixed in time** (current OS release), which is why it cannot be used with LSOA fixed effects.
- **One instrument, one endogenous variable.** The interactions of v3 (PM10 × house, PM10 × low green) are not
  instrumented here.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE}")
    # numbers for the verdict
    for lab, r, m in settings:
        print(lab, "F", round(r["F"], 2), "IV", round(r["coef"] * 100, 3), round(r["se"] * 100, 3), "AR", r["ar"],
              "OLS", ols_cell(m) if m is not None else "")
    print("green coef PM10 fixed: borough", g_b, "lsoa", g_l, "balance", bal, bal_vars)
    print("J", Cg["J"], Cg["pJ"], "mean price", mean_price)


if __name__ == "__main__":
    main()
