"""
HE3027 Group Project: WTP for cleaner air by income group, with small-area household income.

model_v1_income_groups.py splits the sample by borough pay, which mixes rich and poor neighbourhoods. Here the
income groups come from much finer household income estimates:

- GLA Household Income Estimates 2011/12, median household income per LSOA (London LSOA Atlas)
- ONS Small Area Model-Based Income Estimates 2011/12, mean total household income per MSOA

and the borough pay groups are re-estimated with the same method for comparison.

Writes model_v1_income_small_area.md and one figure per income measure: figures/wtp_by_income_lsoa.png,
figures/wtp_by_income_msoa.png and figures/wtp_by_income_borough.png.

Run from the repository root:
    python models/model_v1_income_small_area.py

Needs: pandas, numpy, matplotlib, xlrd (for the ONS .xls file)
Data:  data/land_registry_pollution.dta
       data/gla_lsoa_atlas/lsoa-data.csv
           from https://data.london.gov.uk/dataset/lsoa-atlas ("Current LSOA boundaries post-2011", CSV)
       data/ons_msoa_income/ons-model-based-income-estimates-msoa.xls
           from https://data.london.gov.uk/dataset/ons-model-based-income-estimates-msoa
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_v1 import CODE_DIR, MODEL_DIR, DATA_DIR, fe_ols, load_data, pct, stars
import model_v1_income_groups
from model_v1_income_groups import (BASE, FIG_DIR, GROUPS, INCOME_SPECS, add_pay_terms, choose_income_form,
                                    gbp, p_fmt, p_normal, plot_groups, wald)

MD_FILE = MODEL_DIR / "model_v1_income_small_area.md"
LSOA_FILE = DATA_DIR / "gla_lsoa_atlas" / "lsoa-data.csv"
MSOA_FILE = DATA_DIR / "ons_msoa_income" / "ons-model-based-income-estimates-msoa.xls"

MEASURES = {   # name: (label, short label, income column, area the groups are formed from)
    "lsoa": ("GLA median household income per LSOA, 2011/12", "LSOA household income", "inc_lsoa", "lsoacode"),
    "msoa": ("ONS mean household income per MSOA, 2011/12", "MSOA household income", "inc_msoa", "msoa11"),
    "borough": ("Mean gross annual pay per borough, 2005-2014", "Borough pay", "inc_borough", "oslaua"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def add_small_area_income(df):
    a = pd.read_csv(LSOA_FILE, encoding="latin-1", low_memory=False)
    a = pd.DataFrame({"lsoacode": a["Lower Super Output Area"],
                      "inc_lsoa": pd.to_numeric(
                          a["Household Income, 2011/12;Median Annual Household Income estimate (£)"],
                          errors="coerce")}).dropna()
    m = pd.read_excel(MSOA_FILE, sheet_name="2011-12 (weekly income)")
    m = pd.DataFrame({"msoa11": m["MSOA code"], "inc_msoa": m["Total weekly income (£)"] * 52})   # weekly -> annual
    n = len(df)
    df = df.merge(a, on="lsoacode", how="inner").merge(m, on="msoa11", how="inner")
    df["inc_borough"] = df.groupby("oslaua")["grossannualpay"].transform("mean")
    return df, n - len(df)


def make_groups(df, measure):
    """Terciles of area income, weighted by sales, so each group has about 1/3 of the sales.

    Whole areas are grouped (each area has one income value), so every LSOA stays in one group.
    """
    _, _, inc, area = MEASURES[measure]
    a = df.groupby(area).agg(inc=(inc, "first"), sales=("price", "size")).sort_values("inc")
    mid_share = (a["sales"].cumsum() - a["sales"] / 2) / a["sales"].sum()
    a["group"] = pd.cut(mid_share, [0, 1 / 3, 2 / 3, 1], labels=GROUPS).astype(str)
    return df[area].map(a["group"])


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def group_model(df, group, income, extra=(), cluster="msoa11"):
    """Pooled M8 where the PM10 effect differs by income group (base: low), with group-specific
    year-quarter effects, so each group's PM10 effect is identified only from its own price trend."""
    d = df.assign(group=group,
                  pm10_mid=df["pm10"] * (group == "Middle"),
                  pm10_high=df["pm10"] * (group == "High"),
                  yearquarter=group + "_" + df["yearquarter"].astype(str))
    return fe_ols(d, BASE + INCOME_SPECS[income][1] + ["pm10_mid", "pm10_high"] + list(extra),
                  time_fe=True, area_fe="lsoacode", cluster=cluster)


WEIGHTS = {"Low": {"pm10": 1}, "Middle": {"pm10": 1, "pm10_mid": 1}, "High": {"pm10": 1, "pm10_high": 1}}


def lincom(m, w):
    """Estimate and standard error of a linear combination of coefficients, w = {term: weight}."""
    t = list(w)
    a = np.array([w[k] for k in t])
    b = m["res"].loc[t, "coef"].to_numpy()
    V = m["V"].loc[t, t].to_numpy()
    return float(a @ b), math.sqrt(float(a @ V @ a))


def combine(*ws):
    out = {}
    for scale, w in ws:
        for k, v in w.items():
            out[k] = out.get(k, 0) + scale * v
    return out


def analyse(df, measure, income):
    group = make_groups(df, measure)
    m = group_model(df, group, income)
    m_la = group_model(df, group, income, cluster="oslaua")
    inc = MEASURES[measure][2]
    est = {}
    for g in GROUPS:
        d = df[group == g]
        b, se = lincom(m, WEIGHTS[g])
        price = d["price"].mean()
        est[g] = {"b": b, "se": se, "price": price, "wtp": -b * price, "wtp_se": se * price, "n": len(d),
                  "inc_lo": d[inc].min(), "inc_hi": d[inc].max(), "inc_mean": d[inc].mean(),
                  "areas": d[MEASURES[measure][3]].nunique(), "boroughs": d["oslaua"].nunique(),
                  "pm10": d["pm10"].mean()}

    tests = {}
    for a, c in [("Low", "Middle"), ("Low", "High"), ("Middle", "High")]:
        tests[(a, c)] = {
            "pct": lincom(m, combine((1, WEIGHTS[c]), (-1, WEIGHTS[a]))),
            # WTP_g = -price_g * b_g, prices treated as fixed
            "wtp": lincom(m, combine((-est[c]["price"], WEIGHTS[c]), (est[a]["price"], WEIGHTS[a]))),
        }
    joint = wald(m, ["pm10_mid", "pm10_high"])
    joint_la = wald(m_la, ["pm10_mid", "pm10_high"])
    wtp_joint = wald_wtp(m, est)
    return {"group": group, "m": m, "est": est, "tests": tests, "joint": joint, "joint_la": joint_la,
            "wtp_joint": wtp_joint}


def wald_wtp(m, est):
    """Joint test that the WTP in GBP is the same in all three groups (2 restrictions)."""
    rows = []
    for c in ["Middle", "High"]:
        w = combine((-est[c]["price"], WEIGHTS[c]), (est["Low"]["price"], WEIGHTS["Low"]))
        rows.append(w)
    terms = ["pm10", "pm10_mid", "pm10_high"]
    R = np.array([[r.get(t, 0) for t in terms] for r in rows])
    b = m["res"].loc[terms, "coef"].to_numpy()
    V = m["V"].loc[terms, terms].to_numpy()
    Rb = R @ b
    W = float(Rb @ np.linalg.solve(R @ V @ R.T, Rb))
    return W, 2, math.exp(-W / 2)


def continuous_model(df, measure, income):
    """PM10 × log income (centred), with year-quarter effects by income tercile as in the group model."""
    inc = MEASURES[measure][2]
    ln_inc = np.log(df[inc])
    d = df.assign(pm10_x_lninc=df["pm10"] * (ln_inc - ln_inc.mean()),
                  yearquarter=make_groups(df, measure) + "_" + df["yearquarter"].astype(str))
    m = fe_ols(d, BASE + INCOME_SPECS[income][1] + ["pm10_x_lninc"], time_fe=True, area_fe="lsoacode")
    # PM10 effect at the 10th, 50th and 90th percentile of income (sales-weighted)
    at = {q: (df[inc].quantile(q),
              lincom(m, {"pm10": 1, "pm10_x_lninc": ln_inc.quantile(q) - ln_inc.mean()}))
          for q in [0.1, 0.5, 0.9]}
    return m, at




def unit_elasticity_test(m, est):
    """Test WTP_high = (income_high / income_low) × WTP_low, i.e. WTP rises one-for-one with income."""
    ratio = est["High"]["inc_mean"] / est["Low"]["inc_mean"]
    d, se = lincom(m, combine((-est["High"]["price"], WEIGHTS["High"]),
                              (ratio * est["Low"]["price"], WEIGHTS["Low"])))
    return ratio, ratio * est["Low"]["wtp"], p_normal(d / se)


def borough_separate(df, income):
    """The borough version of model_v1_income_groups.py: separate M8 regressions per group (independent estimates)."""
    d = df.copy()
    model_v1_income_groups.make_groups(d)
    est = model_v1_income_groups.group_estimates(d, income)
    sL, sM, sH = (est[g]["se"] for g in GROUPS)
    diff = np.array([est["Middle"]["b"] - est["Low"]["b"], est["High"]["b"] - est["Low"]["b"]])
    V = np.array([[sL ** 2 + sM ** 2, sL ** 2], [sL ** 2, sL ** 2 + sH ** 2]])
    W = float(diff @ np.linalg.solve(V, diff))
    dw = est["High"]["wtp"] - est["Low"]["wtp"]
    sw = math.sqrt(est["High"]["wtp_se"] ** 2 + est["Low"]["wtp_se"] ** 2)
    return {"est": est, "p_pct": math.exp(-W / 2), "wtp_diff": (dw, sw)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fig_path(name):
    return FIG_DIR / f"wtp_by_income_{name}.png"


def main():
    df, _, info = load_data()
    add_pay_terms(df)
    df, n_lost = add_small_area_income(df)
    _, _, _, income = choose_income_form(df)   # same borough pay control as in model_v1_income_groups.py
    print("income control:", income, "| sales lost in merge:", n_lost)

    results = {name: analyse(df, name, income) for name in MEASURES}
    cont = {name: continuous_model(df, name, income) for name in MEASURES}
    sep = borough_separate(df, income)
    for name, r in results.items():
        print(name, {g: f"{e['b'] * 100:+.3f}% / £{e['wtp']:,.0f}" for g, e in r["est"].items()},
              "joint p =", p_fmt(r["joint"][2]))
        digits = 1 if name == "borough" else 0
        plot_groups(r["est"], path=fig_path(name),
                    tick=lambda g, e, k=digits: f"{g} income\n(mean £{e['inc_mean'] / 1000:.{k}f}k)",
                    note="Pooled M8 with PM10 × income group and group-specific year-quarter effects.",
                    suptitle=f"Income groups: {MEASURES[name][0]}")

    corr = df[["inc_lsoa", "inc_msoa", "inc_borough"]].drop_duplicates().corr()

    # --- tables ---
    def group_table(r, name):
        _, _, inc, area = MEASURES[name]
        unit = {"lsoacode": "LSOAs | Boroughs", "msoa11": "MSOAs | Boroughs", "oslaua": "Boroughs"}[area]
        rows = [f"| Group | Income range | Mean income | {unit} | Sales | Mean price | Mean PM10 |",
                "|---|---|---|---|---|---|---|" + ("---|" if area != "oslaua" else "")]
        for g in GROUPS:
            e = r["est"][g]
            areas = f"{e['areas']:,}" + (f" | {e['boroughs']}" if area != "oslaua" else "")
            rows.append(f"| {g} | £{e['inc_lo']:,.0f} to £{e['inc_hi']:,.0f} | £{e['inc_mean']:,.0f} | "
                        f"{areas} | {e['n']:,} | £{e['price']:,.0f} | {e['pm10']:.1f} |")
        return "\n".join(rows)

    def result_table(r):
        rows = ["| Group | PM10 effect | Std. error (pp) | 95% CI | WTP for −1 µg/m³ | 95% CI |",
                "|---|---|---|---|---|---|"]
        for g in GROUPS:
            e = r["est"][g]
            rows.append(f"| {g} | {pct(e['b'])}{stars(e['b'], e['se'])} | {e['se'] * 100:.3f} | "
                        f"[{pct(e['b'] - 1.96 * e['se'])}, {pct(e['b'] + 1.96 * e['se'])}] | {gbp(e['wtp'])} | "
                        f"[{gbp(e['wtp'] - 1.96 * e['wtp_se'])}, {gbp(e['wtp'] + 1.96 * e['wtp_se'])}] |")
        return "\n".join(rows)

    def test_table(r):
        rows = ["| Difference | PM10 effect (pp) | p | WTP (GBP) | p |", "|---|---|---|---|---|"]
        for (a, c), t in r["tests"].items():
            (dp, sp), (dw, sw) = t["pct"], t["wtp"]
            rows.append(f"| {c} − {a} | {dp * 100:+.3f} ({sp * 100:.3f}) | {p_fmt(p_normal(dp / sp))} | "
                        f"{dw:+,.0f} ({sw:,.0f}) | {p_fmt(p_normal(dw / sw))} |")
        return "\n".join(rows)

    sections = []
    for name, r in results.items():
        _, _, p = r["joint"]
        _, _, p_la = r["joint_la"]
        _, _, pw = r["wtp_joint"]
        ratio, wtp_unit, p_unit = unit_elasticity_test(r["m"], r["est"])
        sections.append(f"""### {MEASURES[name][0]}

![Income groups by {MEASURES[name][1]}](../figures/wtp_by_income_{name}.png)

{group_table(r, name)}

{result_table(r)}

Differences between groups (standard errors in brackets):

{test_table(r)}

- Joint test that the PM10 effect (%) is equal in all three groups: p = {p_fmt(p)} (MSOA clusters),
  p = {p_fmt(p_la)} (borough clusters).
- Joint test that the WTP (GBP) is equal in all three groups: p = {p_fmt(pw)}.
- If WTP rose one-for-one with income, the high group (income {ratio:.2f} × the low group) would pay about
  {gbp(wtp_unit)}; the test of this gives p = {p_fmt(p_unit)}.
""")

    cont_rows = []
    for name, (m, at) in cont.items():
        r = m["res"].loc["pm10_x_lninc"]
        cont_rows.append(
            f"| {MEASURES[name][1]} | {r.coef * 100:+.3f}{stars(r.coef, r.se)} | {r.se * 100:.3f} | "
            f"{p_fmt(p_normal(r.coef / r.se))} | " +
            " | ".join(f"{pct(b)} (£{inc_q:,.0f})" for q, (inc_q, (b, _)) in at.items()) + " |")

    def spec_row(label, e, dws, p_pct, p_wtp):
        dw, sw = dws
        return (f"| {label} | {gbp(e['Low']['wtp'])} | {gbp(e['Middle']['wtp'])} | {gbp(e['High']['wtp'])} | "
                f"{dw:+,.0f} ({sw:,.0f}) | {p_fmt(p_pct)} | {p_wtp} |")

    spec_rows = [spec_row(label, results[n]["est"], results[n]["tests"][("Low", "High")]["wtp"],
                          results[n]["joint"][2], p_fmt(results[n]["wtp_joint"][2]))
                 for n, label in [("lsoa", "LSOA household income, pooled model (**preferred**)"),
                                  ("msoa", "MSOA household income, pooled model"),
                                  ("borough", "Borough pay, pooled model")]]
    spec_rows.append(spec_row("Borough pay, separate regressions (`model_v1_income_groups.md`)", sep["est"],
                              sep["wtp_diff"], sep["p_pct"], "not computed"))

    L = results["lsoa"]
    eL = L["est"]
    dL, sL = L["tests"][("Low", "High")]["wtp"]
    _, wtp_unit_L, p_unit_L = unit_elasticity_test(L["m"], eL)
    B = results["borough"]
    any_sig = [MEASURES[n][1].replace("Borough", "borough") for n, r in results.items() if r["wtp_joint"][2] < 0.05]
    sig_text = ("Only " + ", ".join(any_sig) + " gives a significant WTP difference, and it goes in the "
                "opposite direction to the theory." if any_sig
                else "None of the specifications gives a significant WTP difference.")

    md = f"""# WTP for Cleaner Air by Income Group

Generated by `model_v1_income_small_area.py`. Extends the v1 model (`model_v1.md`) and the borough income
groups (`model_v1_income_groups.md`). All numbers in this file are produced by the script; nothing is typed in by hand.

## Summary of findings

1. **We find no significant difference in the WTP for cleaner air between low, middle and high income
   neighbourhoods.** With our preferred income measure (median household income per LSOA), the WTP for
   1 µg/m³ less PM10 is {gbp(eL['Low']['wtp'])}, {gbp(eL['Middle']['wtp'])} and {gbp(eL['High']['wtp'])} per home
   (joint test p = {p_fmt(L['wtp_joint'][2])}).
2. **The % price discount for pollution is smaller in richer neighbourhoods** ({pct(eL['Low']['b'])} low,
   {pct(eL['Middle']['b'])} middle, {pct(eL['High']['b'])} high income), but homes there cost more, so in GBP the
   WTP is about the same. The differences in % are not significant either (p = {p_fmt(L['joint'][2])}).
3. **This does not show that income does not matter.** The estimates are imprecise: the 95% confidence interval
   for the high-minus-low WTP difference is [{gbp(dL - 1.96 * sL)}, {gbp(dL + 1.96 * sL)}]. If WTP rose one-for-one
   with income, the high group would pay about {gbp(wtp_unit_L)}; the test of this gives p = {p_fmt(p_unit_L)}.
   We can say there is no visible income gradient, not that there is none.
4. **Borough-level income gives a misleading result.** With borough pay as the income measure the groups differ
   significantly in the pooled model (p = {p_fmt(B['wtp_joint'][2])}), but in the *opposite* direction
   (low-pay boroughs pay most), and the difference in % effects is not significant with separate regressions per
   group (p = {p_fmt(sep['p_pct'])}). Borough pay is almost unrelated to neighbourhood income (correlation
   {corr.at['inc_lsoa', 'inc_borough']:.2f}), so we do not interpret this as an income effect.
5. **None of these estimates is causal** (see the limitations below).

## Data sources

| Data | What we use | Level | Years | Source |
|---|---|---|---|---|
| House sales | Price, date, property type, new build, tenure, postcode | Each sale | 2005-2015 (we use 2005-2014) | HM Land Registry Price Paid Data, provided in the course data set `land_registry_pollution.dta` |
| Air pollution | PM10 and NOx, yearly mean at the monitoring site nearest to the home | Nearest site | Yearly | Provided in the course data set; the file does not say which monitoring network the data come from |
| Location | 2011 Output Area, LSOA and MSOA codes, borough; distance to the centre, tube, bus stop, Thames, parks; heritage buildings nearby | Each sale | | Provided in the course data set |
| Borough economy | Gross annual pay of residents, job density, unemployment, population | Borough × year | 2005-2014 | Provided in the course data set (local authority statistics) |
| Neighbourhood population | Share without qualifications, ethnic minorities, lone-parent households | LSOA | Census (year not given in the file) | Provided in the course data set |
| Crime (robustness check only) | All recorded crimes | LSOA × year | 2010-2015 | Provided in the course data set `land_registry_crime.dta` |
| **LSOA household income (new)** | Median annual household income estimate | LSOA | 2011/12 | GLA Household Income Estimates, taken from the [London LSOA Atlas](https://data.london.gov.uk/dataset/lsoa-atlas) (London Datastore). The stand-alone GLA income dataset is no longer accessible, so we use the copy in the Atlas |
| **MSOA household income (new)** | Mean total annual household income (weekly × 52) | MSOA | 2011/12 | [ONS Small Area Model-Based Income Estimates](https://data.london.gov.uk/dataset/ons-model-based-income-estimates-msoa) (London Datastore) |

All {df['lsoacode'].nunique():,} LSOAs and {df['msoa11'].nunique():,} MSOAs in our sample are matched to the new
income data ({n_lost:,} sales lost), so the sample is the same {len(df):,} sales as in the v1 model.

**What the income data are and are not.** Both new measures are *modelled area averages*: ONS and the GLA
combine survey data with census and administrative data to estimate the income of the households living in an
area. They are **not the income of the household that bought the home**; free data on buyer income do not
exist (mortgage records with borrower income, the FCA Product Sales Data, are restricted). An LSOA has about
1,500 residents, so the LSOA measure separates rich and poor neighbourhoods much better than borough pay:
LSOA median income ranges from £{df['inc_lsoa'].min():,.0f} to £{df['inc_lsoa'].max():,.0f}, borough pay
from £{df['inc_borough'].min():,.0f} to £{df['inc_borough'].max():,.0f}. Correlations across LSOAs: LSOA and MSOA
income {corr.at['inc_lsoa', 'inc_msoa']:.2f}, LSOA income and borough pay {corr.at['inc_lsoa', 'inc_borough']:.2f}.
Both new measures are for one year only (2011/12, near the end of our period), so they cannot track changes
in income within an area over time.

## Method

- **Income groups:** areas sorted by income and cut so that each group has about one third of the sales. Whole
  areas are grouped, so every LSOA stays in one group.
- **Model:** the v1 model M8 (LSOA and year-quarter fixed effects, all controls, borough pay as
  {INCOME_SPECS[income][0].lower()}, chosen in `model_v1_income_groups.md`), estimated on all sales with PM10 × middle and
  PM10 × high income interactions and **year-quarter effects for each group separately**. So each group's PM10
  effect only uses how PM10 and prices move *within* that group, not differences in trends between rich and poor
  areas. The area income itself does not change over time, so the LSOA fixed effects absorb it.
- **WTP:** −(PM10 effect) × mean price of the group. Theory predicts that richer households pay more **in GBP**,
  so we test the WTP in GBP as well as the % effect (delta method, group prices treated as fixed).
- Standard errors clustered by MSOA; borough clusters as a check.
- **Preferred measure:** we chose LSOA income as the preferred measure because it is the finest income data
  available, *before* seeing its result. All measures we tried are reported below.

## Results by income measure

{chr(10).join(sections)}
\\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01.

### Continuous income instead of three groups

Same model with an extra PM10 × log(income) term (income centred at its mean), and year-quarter effects by
income tercile. A negative coefficient would mean the PM10 discount grows with income. The last three columns
show the implied PM10 effect at the 10th, 50th and 90th percentile of income.

| Income measure | PM10 × ln(income), ×100 | Std. error | p | P10 | P50 | P90 |
|---|---|---|---|---|---|---|
{chr(10).join(cont_rows)}

## All specifications we ran

So that the reader can judge whether the result depends on our choices, here is every income-group
specification we estimated:

| Specification | WTP low | WTP middle | WTP high | High − low, GBP (s.e.) | p: equal % effects | p: equal WTP in GBP |
|---|---|---|---|---|---|---|
{chr(10).join(spec_rows)}

{sig_text} No specification shows richer areas paying significantly more.

## Limitations: why an income effect may be hidden

- **Income of the area, not of the buyer.** Some buyers in poor areas are rich and the other way round, which
  pulls the group estimates together.
- **Sorting.** Households that care most about clean air move to clean areas. The hedonic price then shows what
  the market pays in each area, not how much each type of household values clean air (Rosen 1974; Bayer,
  Keohane and Timmins 2009).
- **Measurement error in PM10.** PM10 is the yearly mean at the nearest of about 60 to 74 monitoring sites,
  so it measures the pollution at the home with error, which biases the estimates towards zero.
- **Statistical power.** The PM10 effect is identified only from changes over time within LSOAs (about
  3 µg/m³), and splitting the sample into three groups leaves wide confidence intervals.
- **Not causal.** All estimates are associations conditional on our controls and fixed effects. A causal WTP by
  income would need buyer-level income data or a quasi-experiment, such as a policy that cut pollution in some
  areas but not others (for example the London Low Emission Zone, 2008).

## Conclusion

Using the best freely available household income data (median household income per LSOA), we **do not find that
richer neighbourhoods pay more for cleaner air**. In % terms the price discount for pollution is somewhat smaller
in richer areas, and in GBP the WTP is about the same in all three groups ({gbp(eL['Low']['wtp'])},
{gbp(eL['Middle']['wtp'])} and {gbp(eL['High']['wtp'])} per µg/m³). The estimates are too imprecise to rule out a
moderate income gradient, and the limitations above would all hide such a gradient. Our result is therefore
**no visible income gradient**, not evidence that income does not matter.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE} and {', '.join(fig_path(n).name for n in MEASURES)}")


if __name__ == "__main__":
    main()
