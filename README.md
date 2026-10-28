# Hedonic Pricing of Air Quality

HE3027 Housing Economics group project (NTU). Do London house prices reflect air pollution?
Group 12: Biravin Kalathevan, Simon Fröhlich, Celes Ong, Kaveri Goswami and Benjamin Koh Wing Kiat.


**Paper:** link follows once the final report is finished.

## Abstract

We estimate London residents' willingness to pay (WTP) for cleaner air with a hedonic price model of 270,399
residential sales from 2007 to 2014. Pollution is the three-year average of PM10 at the nearest monitoring site.
LSOA and year-quarter fixed effects identify the effect from changes in PM10 within small neighbourhoods
over time. One µg/m³ more PM10 lowers prices by 0.36% on average, about £1,292 per home at the mean price, or
£4.3bn for the London housing stock. The discount is concentrated in flats (−0.42% per µg/m³ in greener postcodes,
−0.65% in the least green third) and is close to zero for houses in greener postcodes. Distance to main roads and road
noise are priced but leave the PM10 effect unchanged. An attempt to instrument PM10 with green space failed because
the instrument is weak, so the estimates are well-controlled associations rather than causal effects.

## Final model (v3)

```math
\begin{aligned}
\ln P_{ijt} = {} & \beta\,\mathrm{PM}^{(3)}_{jt} + \theta_1\,\mathrm{PM}^{(3)}_{jt}\times \mathrm{House}_i + \theta_2\,\mathrm{PM}^{(3)}_{jt}\times \mathrm{LowGreen}_i \\
& + X_i\gamma + R_i\rho + \lambda_1\,\mathrm{Green}_i + \lambda_2\,\mathrm{LowGreen}_i + B_{bt}\delta + \alpha_j + \tau_q + \varepsilon_{ijt}
\end{aligned}
```

- PM10 is the mean of the year of sale and the two years before at the nearest monitoring site.
- The controls are property type, new build, tenure, distances (log and log²), bus access, census shares, the borough
  economy by year, being within 50 m of a main road, and road noise of 65 dB or more.
- α are LSOA fixed effects and τ year-quarter fixed effects. Standard errors are clustered by MSOA.

| Group | Effect of +1 µg/m³ PM10 | WTP per µg/m³ |
|---|---|---|
| All homes (model without interactions) | −0.36%*** | £1,292 |
| Flats, greener postcodes | −0.42%*** | £1,319 |
| Flats, least green third | −0.65%*** | £2,098 |
| Houses, greener postcodes | −0.05% | £209 |
| Houses, least green third | −0.29%** | £1,167 |

\*\* p < 0.05, \*\*\* p < 0.01. Both interactions are significant at 5% in all eight robustness checks, including
LSOA-by-year fixed effects. Full results: [`model_v3.md`](models/model_v3.md).
Total costs: [`total_cost_using_v3.md`](total_cost_using_v3.md).

## Interpretation

- **Buyers pay for cleaner air.** Compared with homes in the same neighbourhood, a home whose PM10 rises by 1 µg/m³
  relative to London as a whole sells for about 0.36% less. At the mean PM10 level this is an elasticity of −0.09.
- **Flats bear most of the discount.** Flat owners have no garden and fewer ways to avoid outdoor air. The discount
  is larger again where there is little green space nearby, but this interaction is the less certain one: it was
  chosen after exploratory tests and does not pass a Bonferroni correction for the twelve interactions tested.
- **PM10 is not a proxy for traffic.** Main roads and noise each lower prices by about 2.5%, but adding them changes
  the PM10 effect by at most 0.02 percentage points.
- **In pounds:** 1 µg/m³ less PM10 across London is worth about £4.3bn (95% CI £2.4bn to £6.2bn), or £150m a year
  at a 3.5% discount rate.
- **Tests without a robust result:** no income gradient in WTP, no non-linearity in PM10, and no effect of NOx or crime
  once PM10 is included ([`null_results.txt`](null_results.txt)). Green space as an instrument for PM10 is too weak
  (first-stage F ≤ 5.6) ([`model_v4.md`](models/model_v4.md)).

## Limitations

- **Measurement.** PM10 is measured at the nearest of 59 to 74 monitoring sites, not at the home. This biases the
  effect towards zero, so the estimates are more likely too small than too large.
- **Causality.** Local shocks that move both pollution and prices could bias β. With borough-by-year fixed effects,
  the effect for flats in greener postcodes falls to −0.16% (significant at 10%).
- **Timing of the added data.** Roads and green space come from current releases and noise from 2012, while the sales
  run from 2007 to 2014. All are measured at the postcode centroid.
- **Marginal WTP.** The effect is identified from small changes (two thirds within ±1 µg/m³), so the totals for
  reaching the WHO guidelines are extrapolations, and they include only costs that buyers perceive.

## Reproducing the results

The data are not in the repository. Put them into `data/` with these names:

| File | Source |
|---|---|
| `land_registry_pollution.dta` | Course data set (HM Land Registry sales with PM10) |
| `gla_lsoa_atlas/lsoa-data.csv` | [London LSOA Atlas](https://data.london.gov.uk/dataset/lsoa-atlas) |
| `os_open_roads/oproad_essh_gb.zip` | [OS Open Roads](https://osdatahub.os.uk/downloads/open/OpenRoads) |
| `defra_road_noise/Road_Lden_London.shp` | [Noise pollution in London](https://data.london.gov.uk/dataset/noise-pollution-in-london-2zwnk) |
| `os_open_greenspace/opgrsp_essh_tq.zip`, `opgrsp_essh_tl.zip` | [OS Open Greenspace](https://osdatahub.os.uk/downloads/open/OpenGreenspace) |

```bash
pip install pandas numpy scipy matplotlib shapely pyproj pyshp
python models/model_v3.py
```

`models/model_v3.py` writes `models/model_v3.md` and builds the road and green-space features on its first run. All model versions are in `models/`:
earlier versions (`model_v0` to `model_v2`), robustness analyses (`model_v1_*`, `model_v2_*`) and the IV attempt
(`model_v4`), each with a `.md` file of its results. A few analyses need extra data, listed at the top of each script.
