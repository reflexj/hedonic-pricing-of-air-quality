"""
HE3027 Group Project: temporary swings in PM10 and the v1 model.

Mean PM10 jumps from 2010 to 2011 and stays high in 2012, then falls again. Buyers who care about air quality
look at the long-run level, so temporary swings in the yearly PM10 value act like measurement error and pull
the estimate towards zero. This script re-estimates the v1 model (M8) with smoothed PM10 measures.

Writes model_v1_pm10_smoothing.md and figures/pm10_smoothing.png.

Run from the repository root:
    python models/model_v1_pm10_smoothing.py

Needs: pandas, numpy, matplotlib
Data:  data/land_registry_pollution.dta
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_v1 import (ACCESS, AMENITY, CLUSTER, CODE_DIR, MODEL_DIR, LAST_YEAR, PROPERTY, SOCIO, fe_ols, load_data,
                      pct, stars)

MD_FILE = MODEL_DIR / "model_v1_pm10_smoothing.md"
FIG_DIR = CODE_DIR / "figures"
FIG_FILE = FIG_DIR / "pm10_smoothing.png"

CONTROLS = PROPERTY + ACCESS + AMENITY + SOCIO     # the M8 controls


# ---------------------------------------------------------------------------
# Data: smoothed PM10 per LSOA and year
# ---------------------------------------------------------------------------
def add_smoothed_pm10(df, full):
    """PM10 is a site-year value (nearest monitoring site), so the mean over the sales in an LSOA-year is the
    PM10 of that LSOA in that year. The panel is built from all years (2005-2015) and only from LSOA-years with
    at least one sale; a moving average is missing if one of its years is missing."""
    panel = full.groupby(["lsoacode", "year"])["pm10"].mean().unstack()
    panel = panel.reindex(columns=range(panel.columns.min(), panel.columns.max() + 1)).T   # years in rows
    measures = {
        "pm10_ma2": panel.rolling(2, min_periods=2).mean(),                 # t-1, t
        "pm10_ma3": panel.rolling(3, min_periods=3).mean(),                 # t-2, t-1, t
        "pm10_ma3c": panel.rolling(3, min_periods=3, center=True).mean(),   # t-1, t, t+1
        "pm10_lag1": panel.shift(1),                                        # t-1
    }
    for name, p in measures.items():
        s = p.T.stack().rename(name).reset_index()
        df = df.merge(s, on=["lsoacode", "year"], how="left")
    return df, panel


# ---------------------------------------------------------------------------
# 2SLS with LSOA fixed effects, year-quarter dummies and MSOA-clustered standard errors
# ---------------------------------------------------------------------------
def fe_iv(d, endog, instr, controls, area_fe="lsoacode", cluster=CLUSTER):
    """Current PM10 instrumented by another PM10 measure. The fixed effects are removed by subtracting LSOA
    means from every variable, then standard 2SLS on the demeaned data."""
    W = pd.concat([d[controls].astype(float),
                   pd.get_dummies(d["yearquarter"], prefix="yq", drop_first=True).astype(float)], axis=1)
    g = d[area_fe]

    def dm(a):
        return a - a.groupby(g).transform("mean")

    W = dm(W)
    W = W.loc[:, W.abs().sum() > 1e-10]
    X = np.column_stack([dm(d[endog].astype(float)), W])
    Z = np.column_stack([dm(d[instr].astype(float)), W])
    y = dm(d["lnprice"].astype(float)).to_numpy()

    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    A_inv = np.linalg.pinv(Xhat.T @ X)
    b = A_inv @ Xhat.T @ y
    e = y - X @ b
    scores = pd.DataFrame(Xhat * e[:, None]).groupby(d[cluster].to_numpy()).sum().to_numpy()
    n, k, G = len(y), X.shape[1], len(scores)
    V = A_inv @ (scores.T @ scores) @ A_inv.T * (G / (G - 1)) * ((n - 1) / (n - k))

    # first stage: PM10 on the instrument and the controls, clustered t-statistic squared = F (one instrument)
    fs = fe_ols(d.assign(lnprice=d[endog]), [instr] + controls, time_fe=True, area_fe=area_fe, cluster=cluster)
    f_coef, f_se = fs["res"].at[instr, "coef"], fs["res"].at[instr, "se"]
    return {"coef": b[0], "se": math.sqrt(V[0, 0]), "n": n, "fs_coef": f_coef, "fs_F": (f_coef / f_se) ** 2}


def m8(d, var):
    m = fe_ols(d.dropna(subset=[var]), [var] + CONTROLS, time_fe=True, area_fe="lsoacode")
    r = m["res"].loc[var]
    return {"coef": r.coef, "se": r.se, "n": m["n"]}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(by_year, specs):
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})

    # left: yearly PM10 vs 3-year average
    a1.axvspan(2010.5, 2012.5, color="#e4e3df", zorder=0)
    a1.plot(by_year.index, by_year["pm10"], color="#2a78d6", linewidth=2, marker="o", markersize=5,
            label="Yearly PM10 (used in M8)", zorder=3)
    ma = by_year["pm10_ma3"].dropna()
    a1.plot(ma.index, ma, color="#eb6834", linewidth=2, marker="o", markersize=5,
            label="3-year average (t-2 to t)", zorder=3)
    a1.set_ylabel("µg/m³ (mean over sales)")
    a1.set_title("PM10 at the nearest monitoring site", loc="left", fontsize=11, color="#0b0b0b")
    a1.set_xticks(by_year.index[::2])
    a1.legend(frameon=False, loc="lower left")
    a1.text(2011.5, a1.get_ylim()[1], "2011-2012", ha="center", va="top", fontsize=9, color="#52514e")

    # right: PM10 effect by specification
    labels = [s[0] for s in specs][::-1]
    vals = [s[1]["coef"] * 100 for s in specs][::-1]
    ci = [1.96 * s[1]["se"] * 100 for s in specs][::-1]
    colors = ["#eb6834" if s[2] else "#2a78d6" for s in specs][::-1]
    y = np.arange(len(specs))
    a2.axvline(0, color="#52514e", linewidth=0.8)
    for i in range(len(specs)):
        a2.errorbar(vals[i], y[i], xerr=ci[i], fmt="o", color=colors[i], ecolor="#0b0b0b", elinewidth=1,
                    capsize=3, markersize=8, zorder=3)
        a2.annotate(f"{vals[i]:+.2f}%", (vals[i], y[i]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#0b0b0b")
    a2.set_yticks(y, labels)
    a2.set_xlabel("% change in price per +1 µg/m³ PM10 (95% CI)")
    a2.set_title("PM10 effect in the v1 model (M8)", loc="left", fontsize=11, color="#0b0b0b")

    for ax in (a1, a2):
        ax.grid(axis="y" if ax is a1 else "x", color="#e4e3df", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Blue: yearly PM10. Orange: smoothed PM10 or instrumented. Standard errors clustered "
             "by MSOA.", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, full, info = load_data()
    df, panel = add_smoothed_pm10(df, full)
    full, _ = add_smoothed_pm10(full, full)

    # --- describe the jump ---
    by_year = full.groupby("year")[["pm10", "pm10_ma3"]].mean()
    sites = full.groupby("year")["pm10"].nunique()          # distinct PM10 values ≈ monitoring sites in use
    change = {y: (panel.loc[y] - panel.loc[y - 1]).dropna() for y in range(panel.index.min() + 1, panel.index.max() + 1)}
    within = df["pm10"] - df.groupby("lsoacode")["pm10"].transform("mean")
    left = within - within.groupby(df["year"]).transform("mean")

    # --- regressions ---
    base = m8(df, "pm10")
    res = {
        "base": base,
        "no1112": m8(df[~df["year"].isin([2011, 2012])], "pm10"),
        "no11": m8(df[df["year"] != 2011], "pm10"),
    }
    for var in ["pm10_ma2", "pm10_ma3", "pm10_ma3c", "pm10_lag1"]:
        s = df.dropna(subset=[var])
        res[var] = m8(s, var)
        res[var + "_cur"] = m8(s, "pm10")                      # current PM10 on the same sample
    s = df.dropna(subset=["pm10_lag1"])
    res["iv_lag"] = fe_iv(s, "pm10", "pm10_lag1", CONTROLS)
    for k, v in res.items():
        print(f"{k:<16} b = {v['coef'] * 100:+.3f}%  se = {v['se'] * 100:.3f}  n = {v['n']:,}")

    specs = [("Yearly PM10 (v1 model M8)", res["base"], False),
             ("Yearly PM10, without 2011-2012", res["no1112"], False),
             ("2-year average (t-1, t)", res["pm10_ma2"], True),
             ("3-year average (t-2 to t)", res["pm10_ma3"], True),
             ("Centred 3-year average (t-1 to t+1)", res["pm10_ma3c"], True),
             ("Yearly PM10, instrumented by last year's", res["iv_lag"], True)]
    plot(by_year, specs)

    # --- markdown ---
    price = df["price"].mean()
    pm10_mean = df["pm10"].mean()

    def row(label, r, same=None):
        c, s = r["coef"], r["se"]
        cur = ""
        if same is not None:
            cur = f"{pct(same['coef'])}{stars(same['coef'], same['se'])}"
        return (f"| {label} | {pct(c)}{stars(c, s)} | {s * 100:.3f} | {cur} | £{-c * price:,.0f} | "
                f"{c * pm10_mean:.3f} | {r['n']:,} |")

    year_rows = "\n".join(
        f"| {y} | {by_year.at[y, 'pm10']:.2f} | "
        + (f"{change[y].mean():+.2f} | {(change[y] > 0).mean() * 100:.0f}% | " if y in change else " | | ")
        + f"{sites.get(y, 0)} | " + (f"{left[df['year'] == y].std():.2f} |" if y <= LAST_YEAR else " |")
        for y in by_year.index)

    iv = res["iv_lag"]
    r3, r3c = res["pm10_ma3"], res["pm10_ma3_cur"]
    ratio = r3["coef"] / r3c["coef"]

    md = f"""# Temporary Swings in PM10 and Model v1

Generated by `model_v1_pm10_smoothing.py`. All numbers are produced by the script.

## The question

Mean PM10 rises sharply from 2010 to 2011, stays high in 2012, and then falls again. A household that cares
about air quality when buying a home looks at the **long-run** air quality of the area, not at one unusual year.
Our v1 model (M8) uses the PM10 of the year of the sale. Does that distort the estimate?

## 1. The 2011-2012 jump in our data

| Year | Mean PM10 (µg/m³) | Mean change per LSOA vs year before | LSOAs where PM10 rose | Monitoring sites in use | Std. dev. of PM10 left after LSOA and year effects |
|---|---|---|---|---|---|
{year_rows}

Means over all sales (2005-2015, prices trimmed). "Monitoring sites in use" counts the distinct PM10 values per
year, which is close to the number of sites that are the nearest site for at least one home. The last column
is the variation that the v1 model actually uses (sample 2005-{LAST_YEAR}).

![PM10 over time and PM10 effect by specification](../figures/pm10_smoothing.png)

- From 2010 to 2011 PM10 rose by {change[2011].mean():.1f} µg/m³ on average and in
  {(change[2011] > 0).mean() * 100:.0f}% of LSOAs, so the jump hit almost all of London at the same time. It lasted
  two years (2011 and 2012), not one.
- The number of monitoring sites drops from {sites[2012]} in 2012 to {sites[2013]} in 2013, so for some homes the
  nearest site changes, which creates further jumps in PM10 that have nothing to do with the air at the home.

## 2. What the sources say about the cause

| Source | What it says |
|---|---|
| London Assembly, written answers to the Mayor, appendices A-C | In the first months of 2011 PM10 in London was 21% to 32% higher than in the same period of 2010, and days above the daily limit were much more frequent at sites across London. |
| City of London, air quality review 2011 (Appendix B) | Eight PM10 episodes in 2011; rainfall was well below average in spring and autumn, which may explain the many episode days. It also says annual levels in the City were broadly similar to 2010. |
| London Borough of Lewisham, air quality progress report 2011 | Episodes of high PM10 occurred across London in 2011. |
| University of Birmingham study of a heavily trafficked London site | Secondary particles, especially nitrate (much of it formed from pollution carried in from the continent), are a major part of PM10 on days above the limit, mainly in early spring and autumn. |
| Defra / NPL Particles Network annual report 2012 | During PM10 episodes in 2012, secondary particles such as nitrate made up most of the PM10 (up to 82% at North Kensington). |
| Atmospheric Chemistry and Physics (2012), European PM measurements | Weather (wind, rain, mixing height, temperature, persistent weather patterns) is a main driver of year-to-year changes in PM across Europe. |

**What we can and cannot confirm.** The sources confirm unusually high PM10 in London in 2011, linked to
episodes, dry weather and particles carried in from the continent. We did **not** find a source that attributes
the higher **annual** mean of 2011 to one European-wide weather event, and the City of London report finds annual
levels similar to 2010. So we describe the jump as *consistent with* weather-driven episodes, not as proven to be
caused by them.

## 3. Why this matters for the regression

- **The London-wide part of the jump does no harm.** The year-quarter fixed effects remove anything that hits all
  of London at the same time.
- **The local part does.** M8 is identified from how one LSOA's PM10 deviates from its own average *after*
  removing the London-wide year effect. Part of that deviation is temporary: weather affecting sites differently,
  one bad year at a site, or a change of the nearest site. If buyers react to long-run air quality, these
  temporary swings do not affect prices. For the regression they are **measurement error**, which biases the
  PM10 coefficient **towards zero**. So the fear is real, but it makes our estimate too *small*, not wrong in sign.

## 4. Solutions tested

| Specification (M8 with all controls, LSOA and year-quarter FE) | PM10 effect | Std. error (pp) | Yearly PM10 on the same sample | WTP per µg/m³ | Elasticity at mean PM10 | Sales |
|---|---|---|---|---|---|---|
{row('Yearly PM10 (v1 model M8)', res['base'])}
{row('Yearly PM10, without 2011 and 2012', res['no1112'])}
{row('Yearly PM10, without 2011', res['no11'])}
{row('2-year average of PM10 (t-1, t)', res['pm10_ma2'], res['pm10_ma2_cur'])}
{row('**3-year average of PM10 (t-2 to t)**', res['pm10_ma3'], res['pm10_ma3_cur'])}
{row('Centred 3-year average (t-1 to t+1)', res['pm10_ma3c'], res['pm10_ma3c_cur'])}
{row("Last year's PM10 instead of this year's", res['pm10_lag1'], res['pm10_lag1_cur'])}
{row("Yearly PM10, instrumented by last year's PM10 (2SLS)", iv, res['pm10_lag1_cur'])}

\\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01. Standard errors clustered by MSOA. WTP = −effect × mean price
(£{price:,.0f}). The column "yearly PM10 on the same sample" re-runs the v1 model on exactly the sales that have
the smoothed measure, so the two can be compared directly: moving averages and lags need PM10 in earlier years,
so they lose the first years of the sample and LSOAs without sales in some years.

**IV first stage:** the coefficient of last year's PM10 on this year's PM10 (with all controls and fixed effects)
is {iv['fs_coef']:.2f}, with F = {iv['fs_F']:.0f} (clustered). An F well above 10 means the instrument is strong.

## 5. Interpretation

- **Dropping 2011-2012** changes the effect only a little ({pct(res['base']['coef'])} → {pct(res['no1112']['coef'])}).
  The specific jump is mostly absorbed by the year fixed effects.
- **Smoothing PM10 makes the effect much larger.** On the same sample the 3-year average gives
  {pct(r3['coef'])} instead of {pct(r3c['coef'])}, about {ratio:.1f} times as large. This is the pattern measurement error
  predicts: averaging removes temporary noise, and the coefficient moves away from zero.
- **The IV points the same way** ({pct(iv['coef'])}). Using last year's PM10 as an instrument keeps only the part of
  this year's PM10 that persists, which removes the temporary noise. Its first stage is strong.
- **Which to use.** We suggest the **3-year average of the sale year and the two years before** as the preferred
  PM10 measure: it only uses information a buyer could know at the time of the sale and matches the idea that
  buyers judge long-run air quality. The centred average uses next year's PM10, which the buyer cannot know, so it
  is only a check. The yearly PM10 estimate in `model_v1.md` most likely understates the effect.
- **What this means for the WTP.** With the 3-year average, 1 µg/m³ less PM10 is worth about
  £{-r3['coef'] * price:,.0f} per home instead of £{-base['coef'] * price:,.0f}, and the elasticity is
  {r3['coef'] * pm10_mean:.3f} instead of {base['coef'] * pm10_mean:.3f}. This is closer to, but still below, the
  -0.20 to -0.35 that Chay and Greenstone (2005) find with a quasi-experiment.
- **Caveats.** The smoothed samples are smaller and start later. A 3-year average still contains some local
  noise and does not fix the basic measurement problem that PM10 comes from the nearest of about 60 to 74
  monitoring sites. A cleaner fix would be weather-adjusted PM10, which needs weather data.

## Sources

- London Assembly: [Written answers to the Mayor, appendices A-C](https://www.london.gov.uk/about-us/londonassembly/meetings/documents/s5002/03e%20Written%20MQT%20Responses%20-%20Appendices%20A-C.pdf)
- City of London: [Air quality review 2011, Appendix B](https://democracy.cityoflondon.gov.uk/documents/s4688/Appendix%20B.pdf)
- London Borough of Lewisham: [Air quality progress report 2011](https://lewisham.gov.uk/-/media/files/imported/progress-20report-202011.pdf?sc_lang=en)
- University of Birmingham: [What are the sources and conditions responsible for exceedences of the 24 h PM10 limit value at a heavily trafficked London site?](https://research.birmingham.ac.uk/en/publications/what-are-the-sources-and-conditions-responsible-for-exceedences-o/)
- Defra / NPL: [Particles Network annual report 2012](https://uk-air.defra.gov.uk/assets/documents/reports/cat05/1312100920_Particles_Network_Annual_report_2012_AS_83.pdf)
- Defra: [Air Pollution in the UK 2011](https://uk-air.defra.gov.uk/assets/documents/annualreport/air_pollution_uk_2011_issue_2.pdf)
- Atmospheric Chemistry and Physics (2012): [Meteorological drivers of PM in Europe, vol. 12, p. 3189](https://acp.copernicus.org/articles/12/3189/2012/acp-12-3189-2012.html)
- European Environment Agency: [Air quality in Europe, 2013 report](https://www.eea.europa.eu/publications/air-quality-in-europe-2013/download) (general background on PM trends; we did not find a statement on 2011 in it)
- Chay, K. and Greenstone, M. (2005), "Does Air Quality Matter? Evidence from the Housing Market",
  *Journal of Political Economy* 113(2).
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE} and {FIG_FILE}")


if __name__ == "__main__":
    main()
