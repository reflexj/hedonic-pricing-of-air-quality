"""
HE3027 Group Project: basic hedonic regressions of house prices on air pollution (PM10),
controlling step by step for crime, distance to the centre and distance to the nearest tube station.

Creates:
    model_v0.md    regression table and short interpretation

Run from the repository root:
    python models/model_v0.py

Needs: pandas, numpy
"""

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent   # models/
CODE_DIR = MODEL_DIR.parent                   # repository root
DATA_DIR = CODE_DIR / "data"
MD_FILE = MODEL_DIR / "model_v0.md"

TRIM = 0.01   # drop the bottom and top 1% of prices (same rule as data_exploration.py)

LABELS = {
    "pm10": "PM10 (µg/m³)",
    "log_crime": "log(1 + crimes in LSOA)",
    "dist_to_cbd": "Distance to centre (km)",
    "tube_km": "Distance to nearest tube station (km)",
    "const": "Constant",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    pol_cols = ["price", "pm10", "year", "lsoacode", "dist_to_cbd", "tube_distnear"]
    df = pd.read_stata(DATA_DIR / "land_registry_pollution.dta", columns=pol_cols,
                       convert_categoricals=False)
    n_raw = len(df)

    # trim prices on the full sample
    lo, hi = df["price"].quantile([TRIM, 1 - TRIM])
    df = df[df["price"].between(lo, hi)].copy()
    n_trimmed = len(df)

    # crime: counts per LSOA and year, available from 2010
    crime = pd.read_stata(DATA_DIR / "land_registry_crime.dta",
                          columns=["lsoacode", "year", "allcrime"], convert_categoricals=False)
    crime = crime.groupby(["lsoacode", "year"], as_index=False).first()
    df = df.merge(crime, on=["lsoacode", "year"], how="left")

    df["lnprice"] = np.log(df["price"])
    df["log_crime"] = np.log1p(df["allcrime"])
    df["tube_km"] = df["tube_distnear"] / 1000

    # common sample: all variables available (crime only from 2010)
    vars_needed = ["lnprice", "pm10", "log_crime", "dist_to_cbd", "tube_km"]
    df = df.dropna(subset=vars_needed)

    info = {"n_raw": n_raw, "n_trimmed": n_trimmed, "lo": lo, "hi": hi,
            "n_final": len(df), "years": (int(df["year"].min()), int(df["year"].max()))}
    return df, info


# ---------------------------------------------------------------------------
# OLS with robust (HC1) standard errors
# ---------------------------------------------------------------------------
def ols(df, y, xvars):
    X = np.column_stack([np.ones(len(df))] + [df[v].to_numpy(float) for v in xvars])
    Y = df[y].to_numpy(float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b
    n, k = X.shape
    meat = (X * e[:, None]).T @ (X * e[:, None])
    V = XtX_inv @ meat @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((Y - Y.mean()) @ (Y - Y.mean()))
    names = ["const"] + xvars
    return {"coef": dict(zip(names, b)), "se": dict(zip(names, se)), "n": n, "r2": r2}


def stars(coef, se):
    t = abs(coef / se)
    return "***" if t > 2.576 else "**" if t > 1.96 else "*" if t > 1.645 else ""


def make_table(models, order):
    header = "| | " + " | ".join(f"({i + 1})" for i in range(len(models))) + " |"
    sep = "|---|" + "---|" * len(models)
    rows = [header, sep]
    for v in order:
        cells, ses = [], []
        for m in models:
            if v in m["coef"]:
                c, s = m["coef"][v], m["se"][v]
                cells.append(f"{c:.5f}{stars(c, s)}")
                ses.append(f"({s:.5f})")
            else:
                cells.append("")
                ses.append("")
        rows.append(f"| {LABELS[v]} | " + " | ".join(cells) + " |")
        rows.append("| | " + " | ".join(ses) + " |")
    rows.append("| Observations | " + " | ".join(f"{m['n']:,}" for m in models) + " |")
    rows.append("| R² | " + " | ".join(f"{m['r2']:.3f}" for m in models) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, info = load_data()

    specs = [
        ["pm10"],
        ["pm10", "log_crime"],
        ["pm10", "log_crime", "dist_to_cbd"],
        ["pm10", "log_crime", "dist_to_cbd", "tube_km"],
    ]
    models = [ols(df, "lnprice", xs) for xs in specs]
    for i, m in enumerate(models, 1):
        print(f"({i}) b_pm10 = {m['coef']['pm10']: .5f}  se = {m['se']['pm10']:.5f}  R2 = {m['r2']:.3f}")

    table = make_table(models, ["pm10", "log_crime", "dist_to_cbd", "tube_km", "const"])
    b_first, b_last = models[0]["coef"]["pm10"], models[-1]["coef"]["pm10"]
    se_last = models[-1]["se"]["pm10"]
    last = models[-1]["coef"]

    def pct(b):
        return f"{abs(b) * 100:.2f}% {'higher' if b > 0 else 'lower'}"

    md = f"""# Basic Regressions: House Prices and Air Pollution

Generated by `model_v0.py`.

## Model

ln(price) = a + b · PM10 + controls + e

Controls are added one at a time:

1. PM10 only
2. \\+ crime: log(1 + all crimes in the LSOA in that year)
3. \\+ distance to the centre (km)
4. \\+ distance to the nearest tube station (km)

Estimated by OLS with robust (heteroskedasticity-consistent) standard errors.

## Sample

- Prices are trimmed: the bottom and top 1% are dropped (below GBP {info['lo']:,.0f} and above GBP {info['hi']:,.0f}),
  leaving {info['n_trimmed']:,} of {info['n_raw']:,} sales.
- Crime data only exist from 2010, so all four regressions use the sales from
  **{info['years'][0]} to {info['years'][1]}** with complete data: **{info['n_final']:,} sales**.
  Using the same sample in every column makes the coefficients comparable.

## Results

{table}

Robust standard errors in brackets. \\* p < 0.10, \\*\\* p < 0.05, \\*\\*\\* p < 0.01.

## Interpretation

- **PM10 only (1):** one extra µg/m³ of PM10 goes together with a {pct(b_first)} price.
- **All controls (4):** one extra µg/m³ of PM10 goes together with a {pct(b_last)} price
  (standard error {se_last * 100:.2f} percentage points).
- **Distance to the centre:** each extra km from the centre goes together with a {pct(last['dist_to_cbd'])} price.
- **Distance to the tube:** each extra km from the nearest tube station goes together with a {pct(last['tube_km'])} price.
- **Crime:** a 1% increase in crime goes together with a {abs(last['log_crime']):.3f}% {'higher' if last['log_crime'] > 0 else 'lower'} price.

## Limitations

These regressions do not yet control for housing characteristics (property type, tenure, new build),
neighbourhood characteristics, or time and area fixed effects. Prices rose strongly over time while
PM10 changed from year to year, and both vary across areas for many other reasons, so the PM10
coefficient is still likely to be biased. These are starting points, not the final WTP estimate.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE}")


if __name__ == "__main__":
    main()
