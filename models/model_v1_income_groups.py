"""
HE3027 Group Project: gross annual income and the willingness to pay (WTP) for cleaner air.

Builds on the v1 model (M8: LSOA and year-quarter fixed effects, MSOA-clustered standard errors).

Part 1: how should gross annual pay enter M8? Linear, log, quadratic or cubic.
Part 2: split the boroughs into low, middle and high income groups and test whether the PM10
        effect (and the implied WTP) differs between them.

Writes model_v1_income_groups.md and figures/wtp_by_income_group.png.

Run from the repository root:
    python models/model_v1_income_groups.py

Needs: pandas, numpy, matplotlib
Data:  data/land_registry_pollution.dta
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_v1 import (ACCESS, AMENITY, CODE_DIR, MODEL_DIR, PROPERTY, SOCIO, fe_ols, load_data, pct, stars)

MD_FILE = MODEL_DIR / "model_v1_income_groups.md"
FIG_DIR = CODE_DIR / "figures"
FIG_FILE = FIG_DIR / "wtp_by_income_group.png"

BASE = ["pm10"] + PROPERTY + ACCESS + AMENITY + [v for v in SOCIO if v != "pay_k"]   # M8 without pay
GROUPS = ["Low", "Middle", "High"]
COLORS = {"Low": "#2a78d6", "Middle": "#eb6834", "High": "#1baf7a"}   # validated categorical slots 1-3

INCOME_SPECS = {   # name: (description, income terms)
    "none": ("No income control", []),
    "linear": ("Linear: pay", ["pay_c"]),
    "log": ("Log: ln(pay)", ["ln_pay"]),
    "quadratic": ("Quadratic: pay + pay²", ["pay_c", "pay_c2"]),
    "cubic": ("Cubic: pay + pay² + pay³", ["pay_c", "pay_c2", "pay_c3"]),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def p_normal(t):
    """Two-sided p-value of a t/z statistic (normal approximation, large sample)."""
    return math.erfc(abs(t) / math.sqrt(2))


def wald(m, terms):
    """Wald test that all `terms` are zero. Returns (W, df, p). p is exact for df = 1 and df = 2."""
    b = m["res"].loc[terms, "coef"].to_numpy()
    V = m["V"].loc[terms, terms].to_numpy()
    W = float(b @ np.linalg.solve(V, b))
    q = len(terms)
    p = p_normal(math.sqrt(W)) if q == 1 else math.exp(-W / 2) if q == 2 else float("nan")
    return W, q, p


def bic(m):
    """BIC up to a constant that is the same for all models on the same sample."""
    return m["n"] * math.log(1 - m["r2"]) + m["k"] * math.log(m["n"])


def gbp(x):
    return f"−£{-x:,.0f}" if x < 0 else f"£{x:,.0f}"


def p_fmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def m8(d, income, cluster="msoa11"):
    return fe_ols(d, BASE + INCOME_SPECS[income][1], time_fe=True, area_fe="lsoacode", cluster=cluster)


# ---------------------------------------------------------------------------
# Part 1: functional form of income
# ---------------------------------------------------------------------------
def choose_income_form(df):
    fits = {name: m8(df, name) for name in INCOME_SPECS}
    fits_la = {name: m8(df, name, cluster="oslaua") for name in INCOME_SPECS}   # pay varies by borough-year

    # each non-linear step is kept only if its extra term is significant at 5% with borough
    # clustering (the conservative choice, because pay only varies by borough and year) AND it lowers the BIC
    chosen, steps = "linear", []
    for prev, new, term in [("linear", "quadratic", "pay_c2"), ("quadratic", "cubic", "pay_c3")]:
        _, _, p_ms = wald(fits[new], [term])
        _, _, p_la = wald(fits_la[new], [term])
        better_bic = bic(fits[new]) < bic(fits[prev])
        keep = p_la < 0.05 and better_bic and chosen == prev
        steps.append((prev, new, term, p_ms, p_la, better_bic, keep))
        if keep:
            chosen = new
    if chosen == "linear" and bic(fits["log"]) < bic(fits["linear"]):   # same number of terms: BIC decides
        chosen = "log"
    return fits, fits_la, steps, chosen


# ---------------------------------------------------------------------------
# Part 2: income groups
# ---------------------------------------------------------------------------
def make_groups(df):
    """Assign whole boroughs to terciles of borough mean pay, so that each group has about 1/3 of the sales.

    Boroughs (not borough-years) are grouped, so every LSOA stays in one group for the whole period and the
    LSOA fixed effects work as in M8.
    """
    b = df.groupby("oslaua").agg(name=("la_name", "first"), pay=("grossannualpay", "mean"),
                                 sales=("price", "size")).sort_values("pay")
    mid_share = (b["sales"].cumsum() - b["sales"] / 2) / b["sales"].sum()   # middle of each borough's share
    b["group"] = pd.cut(mid_share, [0, 1 / 3, 2 / 3, 1], labels=GROUPS).astype(str)
    df["group"] = df["oslaua"].map(b["group"])
    return b


def group_estimates(df, income):
    out = {}
    for g in GROUPS:
        d = df[df["group"] == g]
        m = m8(d, income)
        b, se = m["res"].at["pm10", "coef"], m["res"].at["pm10", "se"]
        price = d["price"].mean()
        out[g] = {"m": m, "b": b, "se": se, "price": price, "wtp": -b * price, "wtp_se": se * price,
                  "n": m["n"], "pay": d["grossannualpay"].mean(), "pm10": d["pm10"].mean(),
                  "pm10_within": (d["pm10"] - d.groupby("lsoacode")["pm10"].transform("mean")).std(),
                  "boroughs": d["oslaua"].nunique(), "msoa": d["msoa11"].nunique()}
    return out


def interaction_model(df, income, cluster="msoa11"):
    """Pooled M8 where only the PM10 effect differs by group (base: low income)."""
    d = df.copy()
    d["pm10_mid"] = d["pm10"] * (d["group"] == "Middle")
    d["pm10_high"] = d["pm10"] * (d["group"] == "High")
    return fe_ols(d, BASE + INCOME_SPECS[income][1] + ["pm10_mid", "pm10_high"],
                  time_fe=True, area_fe="lsoacode", cluster=cluster)


def diff_test(a, b):
    """Groups are disjoint sets of boroughs and MSOAs sit inside boroughs, so the estimates are independent."""
    d = b["b"] - a["b"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return d, se, p_normal(d / se)


def plot_groups(est, path=FIG_FILE, tick=lambda g, e: f"{g} income\n(£{e['pay'] / 1000:.1f}k pay)",
                note="v1 model (M8) estimated separately per income group.", suptitle=None):
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(GROUPS))
    panels = [
        ("PM10 effect on price", "% change in price per +1 µg/m³ PM10",
         [est[g]["b"] * 100 for g in GROUPS], [1.96 * est[g]["se"] * 100 for g in GROUPS], "{:+.2f}%"),
        ("WTP for 1 µg/m³ less PM10 (at group mean price)", "GBP per home",
         [est[g]["wtp"] for g in GROUPS], [1.96 * est[g]["wtp_se"] for g in GROUPS], "£{:,.0f}"),
    ]
    for ax, (title, ylabel, vals, ci, label) in zip(axes, panels):
        for i, g in enumerate(GROUPS):
            ax.bar(x[i], vals[i], width=0.55, color=COLORS[g], zorder=2)
            ax.errorbar(x[i], vals[i], yerr=ci[i], color="#0b0b0b", capsize=5, linewidth=1.2, zorder=3)
            off = ci[i] if vals[i] >= 0 else -ci[i]
            ax.annotate(label.format(vals[i]), (x[i], vals[i] + off), xytext=(0, 4 if vals[i] >= 0 else -4),
                        textcoords="offset points", ha="center", va="bottom" if vals[i] >= 0 else "top",
                        color="#0b0b0b", fontsize=9)
        ax.axhline(0, color="#52514e", linewidth=0.8, zorder=1)
        ax.set_xticks(x, [tick(g, est[g]) for g in GROUPS])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))   # room for the value labels
    fig.text(0.01, 0.01, f"{note} Bars: point estimate; whiskers: 95% confidence interval (MSOA-clustered).",
             fontsize=8, color="#52514e")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="bold", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def add_pay_terms(df):
    pay_mean = df["pay_k"].mean()
    df["pay_c"] = df["pay_k"] - pay_mean            # centred, so pay, pay² and pay³ are less collinear
    df["pay_c2"] = df["pay_c"] ** 2
    df["pay_c3"] = df["pay_c"] ** 3
    df["ln_pay"] = np.log(df["grossannualpay"])
    return pay_mean


def main():
    df, _, info = load_data()
    pay_mean = add_pay_terms(df)

    # --- part 1 ---
    fits, fits_la, steps, chosen = choose_income_form(df)
    for name, m in fits.items():
        r = m["res"].loc["pm10"]
        print(f"{name:<10} b_pm10 = {r.coef: .5f} (se {r.se:.5f})  R2 = {m['r2']:.5f}  BIC = {bic(m):.1f}")
    print("chosen income form:", chosen)

    # --- part 2 ---
    boroughs = make_groups(df)
    est = group_estimates(df, chosen)
    inter = interaction_model(df, chosen)
    inter_la = interaction_model(df, chosen, cluster="oslaua")
    for g in GROUPS:
        print(f"{g:<7} b_pm10 = {est[g]['b']: .5f} (se {est[g]['se']:.5f})  WTP = {est[g]['wtp']:,.0f}  "
              f"n = {est[g]['n']:,}")
    plot_groups(est)

    # --- write markdown ---
    ref = fits["linear"]

    def pay_row(name):
        m, m_la = fits[name], fits_la[name]
        terms = INCOME_SPECS[name][1]
        b, s = m["res"].at["pm10", "coef"], m["res"].at["pm10", "se"]
        if terms:
            _, q, p = wald(m, terms)
            _, _, p_la = wald(m_la, terms)
            test = f"{p_fmt(p)} / {p_fmt(p_la)}" if q <= 2 else "see below"
        else:
            test = ""
        d_bic = bic(m) - bic(ref)
        mark = " **(chosen)**" if name == chosen else ""
        return (f"| {INCOME_SPECS[name][0]}{mark} | {pct(b)}{stars(b, s)} | {s * 100:.3f} | "
                f"{m['r2']:.5f} | {d_bic:+.1f} | {test} |")

    def coef_rows(name):
        m = fits[name]
        return "\n".join(f"| {name} | `{t}` | {m['res'].at[t, 'coef']:+.6f} | {m['res'].at[t, 'se']:.6f} | "
                         f"{fits_la[name]['res'].at[t, 'se']:.6f} |" for t in INCOME_SPECS[name][1])

    step_rows = "\n".join(
        f"| {prev} → {new} | `{term}` | {p_fmt(p_ms)} | {p_fmt(p_la)} | {'Yes' if bb else 'No'} | "
        f"{'Yes' if keep else 'No'} |" for prev, new, term, p_ms, p_la, bb, keep in steps)

    b_none = fits["none"]["res"].at["pm10", "coef"]
    b_lin = fits["linear"]["res"].at["pm10", "coef"]
    b_ch = fits[chosen]["res"].at["pm10", "coef"]
    pm10_range = [fits[n]["res"].at["pm10", "coef"] for n in INCOME_SPECS]

    # part 2 numbers
    grp_rows = "\n".join(
        f"| {g} | {est[g]['boroughs']} | {est[g]['msoa']} | {est[g]['n']:,} | £{est[g]['pay']:,.0f} | "
        f"£{est[g]['price']:,.0f} | {est[g]['pm10']:.1f} | {est[g]['pm10_within']:.2f} |" for g in GROUPS)
    borough_lists = "\n".join(
        f"- **{g}:** " + ", ".join(f"{r.name} (£{r.pay / 1000:.1f}k)" for r in boroughs[boroughs['group'] == g]
                                   .itertuples()) for g in GROUPS)
    res_rows = "\n".join(
        f"| {g} | {pct(est[g]['b'])}{stars(est[g]['b'], est[g]['se'])} | {est[g]['se'] * 100:.3f} | "
        f"[{pct(est[g]['b'] - 1.96 * est[g]['se'])}, {pct(est[g]['b'] + 1.96 * est[g]['se'])}] | "
        f"{gbp(est[g]['wtp'])} | [{gbp(est[g]['wtp'] - 1.96 * est[g]['wtp_se'])}, "
        f"{gbp(est[g]['wtp'] + 1.96 * est[g]['wtp_se'])}] | {est[g]['b'] * est[g]['pm10']:.3f} | "
        f"{est[g]['n']:,} |" for g in GROUPS)

    pairs = [("Low", "Middle"), ("Low", "High"), ("Middle", "High")]
    diffs = {(a, c): diff_test(est[a], est[c]) for a, c in pairs}
    diff_rows = "\n".join(f"| {c} − {a} | {d * 100:+.3f} pp | {s * 100:.3f} | {p_fmt(p)} |"
                          for (a, c), (d, s, p) in diffs.items())

    def irow(label, term):
        r1, r2 = inter["res"].loc[term], inter_la["res"].loc[term]
        return (f"| {label} | {r1.coef * 100:+.3f}{stars(r1.coef, r1.se)} | {r1.se * 100:.3f} | "
                f"{p_fmt(p_normal(r1.coef / r1.se))} | {r2.se * 100:.3f} | {p_fmt(p_normal(r2.coef / r2.se))} |")

    W, _, p_joint = wald(inter, ["pm10_mid", "pm10_high"])
    W_la, _, p_joint_la = wald(inter_la, ["pm10_mid", "pm10_high"])
    significant = p_joint < 0.05 and p_joint_la < 0.05
    any_pair_sig = [f"{c} vs {a}" for (a, c), (_, _, p) in diffs.items() if p < 0.05]
    order = sorted(GROUPS, key=lambda g: est[g]["wtp"], reverse=True)

    md = f"""# Income and the Willingness to Pay for Cleaner Air

Generated by `model_v1_income_groups.py`. Builds on the v1 model M8 from `model_v1.md`
(ln price on PM10, property, accessibility, amenity and socioeconomic controls, LSOA and year-quarter fixed
effects, standard errors clustered by MSOA). Same sample: {info['n_final']:,} sales, 2005-2014,
{info['n_la']} London boroughs.

**Income variable:** `grossannualpay`, the mean gross annual pay of residents of the borough in a given
year (ONS). It is the only income measure in the data. It varies by **borough and year**, not by
household or LSOA, so with LSOA fixed effects it is identified only from changes in borough pay over time.

## Part 1: How should income enter the model?

The v1 model already contains pay linearly (`pay_k`). Here we compare that with no income control
and with three non-linear forms. Pay is centred at its mean (£{pay_mean * 1000:,.0f}) before taking
squares and cubes, so `pay_c` = pay in £1,000 minus {pay_mean:.2f}.

| Income control in M8 | PM10 effect | Std. error (pp) | R² | ΔBIC vs linear | p (income terms = 0): MSOA / borough clusters |
|---|---|---|---|---|---|
{chr(10).join(pay_row(n) for n in INCOME_SPECS)}

ΔBIC < 0 means a better fit after penalising extra parameters. The test column is the Wald test that all
income terms in that row are zero (for the cubic with 3 terms, see the step-by-step tests).

**Income coefficients:**

| Model | Term | Coefficient | Std. error (MSOA) | Std. error (borough) |
|---|---|---|---|---|
{coef_rows('linear')}
{coef_rows('log')}
{coef_rows('quadratic')}
{coef_rows('cubic')}

**Step-by-step choice.** A non-linear term is kept only if it is significant at 5% with standard errors
clustered by **borough** (the conservative choice, because pay only varies by borough and year; with MSOA
clusters its standard error is too small) **and** it lowers the BIC. Log and linear have the same number
of terms, so the BIC decides between them.

| Step | Added term | p (MSOA clusters) | p (borough clusters) | Lower BIC? | Keep? |
|---|---|---|---|---|---|
{step_rows}

**Choice: {chosen} ({INCOME_SPECS[chosen][0].split(': ')[1]}).**
The PM10 effect hardly depends on how income enters: it is between {pct(max(pm10_range))} and
{pct(min(pm10_range))} in all five rows ({pct(b_none)} without income, {pct(b_lin)} linear,
{pct(b_ch)} with the chosen form). Income is a borough-year variable and the fixed effects already absorb most
of the differences in income between places, so its functional form matters little for the PM10 estimate.
Pay is skewed (Tower Hamlets, at about £44k, is far above all other boroughs), so the higher-order terms may be
shaped by a few boroughs; read the curve as a flexible control, not as a structural income effect.
All income-group models below use the chosen form.

## Part 2: Does the WTP differ by income group?

### Groups

Each borough is put into one group by its mean gross annual pay over 2005-2014, so that each group has
about one third of the sales. Whole boroughs are grouped (not borough-years), so every LSOA stays in the
same group for the whole period and the LSOA fixed effects work as in M8.

| Group | Boroughs | MSOAs | Sales | Mean pay | Mean price | Mean PM10 | Std. dev. of PM10 within LSOA |
|---|---|---|---|---|---|---|---|
{grp_rows}

The last column is the variation that identifies the PM10 effect (PM10 changes over time within an LSOA).

{borough_lists}

### Results

M8 (with the chosen income control) estimated separately for each group:

![PM10 effect and WTP by income group](../figures/wtp_by_income_group.png)

| Group | PM10 effect | Std. error (pp) | 95% CI | WTP for −1 µg/m³ | 95% CI | Elasticity at mean PM10 | Sales |
|---|---|---|---|---|---|---|---|
{res_rows}

WTP = −(PM10 coefficient) × mean price of the group: what a home buyer pays for a home with 1 µg/m³ less PM10.
Elasticity = % change in price for a 1% change in PM10, at the group's mean PM10.
\\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01.

**Why each group estimate is smaller than the pooled {pct(b_ch)}:** a separate regression per group also gives
each group its own year-quarter effects and its own control coefficients. With common year-quarter effects
(Test 2 below) the group effects are larger again ({pct(inter['res'].at['pm10', 'coef'])} low,
{pct(inter['res'].at['pm10', 'coef'] + inter['res'].at['pm10_mid', 'coef'])} middle,
{pct(inter['res'].at['pm10', 'coef'] + inter['res'].at['pm10_high', 'coef'])} high). So part of the pooled
estimate comes from differences in PM10 and price *trends between groups of boroughs*, the same pattern as
with borough-by-year fixed effects in `model_v1.md` (the effect falls to about −0.07%).

### Are the groups significantly different?

**Test 1: pairwise differences between the separate regressions** (in percentage points of the PM10 effect).
The groups contain different boroughs, and the MSOA clusters lie inside boroughs, so the three
estimates are independent and the standard error of a difference is √(se₁² + se₂²).

| Difference | Estimate | Std. error | p-value |
|---|---|---|---|
{diff_rows}

**Test 2: one pooled M8 with PM10 × group interactions** (base group: low income). Here only the PM10
effect differs by group; all other coefficients are common.

| Term | Coefficient ×100 | Std. error (MSOA) | p | Std. error (borough, 32 clusters) | p |
|---|---|---|---|---|---|
{irow('PM10 (low income)', 'pm10')}
{irow('PM10 × middle income', 'pm10_mid')}
{irow('PM10 × high income', 'pm10_high')}

Joint test that the PM10 effect is the same in all three groups (both interactions = 0):
W = {W:.2f}, p = {p_fmt(p_joint)} (MSOA clusters); W = {W_la:.2f}, p = {p_fmt(p_joint_la)} (borough clusters).

## Conclusion

- **Income control:** the {chosen} form fits best by our rule ({INCOME_SPECS[chosen][0].split(': ')[1]}); the PM10
  effect is almost the same with any form of income ({pct(max(pm10_range))} to {pct(min(pm10_range))}).
- **WTP by income group:** the WTP for 1 µg/m³ less PM10 is £{est['Low']['wtp']:,.0f} in the low,
  £{est['Middle']['wtp']:,.0f} in the middle and £{est['High']['wtp']:,.0f} in the high income group
  (highest: {order[0].lower()}, lowest: {order[-1].lower()}).
- **Significance:** the joint test {'rejects' if p_joint < 0.05 else 'does not reject'} equal PM10 effects with MSOA
  clusters (p = {p_fmt(p_joint)}) and {'rejects' if p_joint_la < 0.05 else 'does not reject'} them with borough clusters
  (p = {p_fmt(p_joint_la)}). Pairwise differences significant at 5%: {', '.join(any_pair_sig) if any_pair_sig else 'none'}.
  {'The WTP for clean air **differs significantly** between income groups.' if significant else
   'So we **cannot conclude** that the WTP for clean air differs between income groups, at least not robustly.'}
- **Caveats:** the ranking is not monotonic in income (middle is lowest), which also points to noise rather than
  a clear income gradient. Income is measured at the borough level, so the groups compare rich and poor *boroughs*, not
  rich and poor buyers. Each group has only about ten boroughs and PM10 comes from a small number of monitoring
  sites, so the group estimates are much less precise than the pooled estimate. Group differences in the
  WTP in GBP also reflect differences in house prices, not only in the % effect. Borough pay is resident pay,
  which ranks some boroughs unexpectedly (e.g. Kensington and Chelsea is in the middle group).
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE} and {FIG_FILE}")


if __name__ == "__main__":
    main()
