# HE3027 Group Project: Air Quality and House Prices in London

## Data

The data files are not on GitHub (too large). Put them into the `data/` folder:

```
Code/
└── data/
    ├── land_registry_pollution.dta
    ├── land_registry_crime.dta
    ├── land_registry_tube.dta
    ├── gla_lsoa_atlas/lsoa-data.csv                              (only for model_v1_income_small_area.py, model_v2.py)
    ├── ons_msoa_income/ons-model-based-income-estimates-msoa.xls  (only for model_v1_income_small_area.py)
    ├── os_open_roads/oproad_essh_gb.zip                          (only for model_v2_roads.py, keep zipped)
    ├── defra_road_noise/Road_Lden_London.shp (+ .dbf .shx .prj)  (only for model_v2_roads.py, unzipped)
    ├── road_features.csv                                         (created by model_v2_roads.py on its first run)
    ├── epc_addresscloud/<borough>.parquet (33 files)             (only for model_v1_energy_efficiency.py)
    ├── os_codepoint_open/codepo_gb.zip                           (only for model_v1_energy_efficiency.py, keep zipped)
    ├── epc_postcode.csv + epc_postcode_info.json                 (created by model_v1_energy_efficiency.py on its first run)
    ├── os_open_greenspace/opgrsp_essh_tq.zip, opgrsp_essh_tl.zip  (only for model_v1_green_space.py, keep zipped)
    ├── green_features.csv                                        (created by model_v1_green_space.py on its first run; also used by model_v3.py)
    └── london_boundaries/statistical-gis-boundaries-london.zip   (only for air_pollution_maps.py, keep zipped)
```

The two income files are free downloads from the London Datastore:
[LSOA Atlas](https://data.london.gov.uk/dataset/lsoa-atlas) ("Current LSOA boundaries post-2011", CSV) and
[ONS Model-Based Income Estimates, MSOA](https://data.london.gov.uk/dataset/ons-model-based-income-estimates-msoa).

The road data are also free (Open Government Licence):
[OS Open Roads](https://osdatahub.os.uk/downloads/open/OpenRoads) (ESRI Shapefile, GB, about 600 MB; only the
London grid squares TQ and TL are read) and the Defra road noise map 2012 from the London Datastore,
[Noise pollution in London](https://data.london.gov.uk/dataset/noise-pollution-in-london-2zwnk) (`Road_Lden_London.zip`).

The energy efficiency data are free as well (Open Government Licence, no account needed):
EPC ratings from the copy of the EPC register by [Addresscloud on Source Cooperative](https://source.coop/addresscloud/epc)
(folder `geoparquet-local-authority`, one file per London borough, e.g.
`https://data.source.coop/addresscloud/epc/geoparquet-local-authority/Camden.parquet`; names with `_` instead of
spaces) and the postcode centroids [OS Code-Point Open](https://osdatahub.os.uk/downloads/open/CodePointOpen)
(CSV, GB, about 15 MB).

The green-space data are [OS Open Greenspace](https://osdatahub.os.uk/downloads/open/OpenGreenspace) (ESRI Shapefile,
the 100 km squares TQ and TL, about 5 MB and 2 MB, Open Government Licence).

The London boundaries for the maps are the
[Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london)
(`statistical-gis-boundaries-london.zip`, about 27 MB, London Datastore, Open Government Licence).

The file names must stay exactly like this, otherwise the code will not find them.

## Run

The file names show which model a script belongs to: `model_v0` (first regressions), `model_v1` (M8 with
the controls step by step) and `model_v1_*` (analyses built on it), `model_v2` and `model_v2_*`, and `model_v3`
(the complete final model).

```
pip install pandas numpy matplotlib xlrd scipy shapely pyproj pyshp pyarrow
python data_exploration.py
python model_v0.py
python model_v1.py
python model_v1_non-linear.py
python model_v1_income_groups.py
python model_v1_income_small_area.py
python model_v1_pm10_smoothing.py
python model_v1_energy_efficiency.py
python model_v1_green_space.py
python model_v2.py
python model_v2_roads.py
python model_v3.py
python air_pollution_maps.py
```

- `data_exploration.py` creates `data_exploration.md` and the graphs in `figures/`.
- `model_v0.py` creates `model_v0.md` with the regression table.
- `model_v1.py` creates `model_v1.md` (controls step by step, v1 model M8, robustness checks).
- `model_v1_non-linear.py` creates `model_v1_non-linear.md` (non-linear distance controls; needs scipy).
- `model_v1_income_groups.py` creates `model_v1_income_groups.md` and `figures/wtp_by_income_group.png` (functional form of income, WTP by low/middle/high income boroughs).
- `model_v1_income_small_area.py` creates `model_v1_income_small_area.md` and one graph per income measure, `figures/wtp_by_income_{lsoa,msoa,borough}.png` (WTP by income group with LSOA and MSOA household income, data sources, all specifications).
- `model_v1_pm10_smoothing.py` creates `model_v1_pm10_smoothing.md` and `figures/pm10_smoothing.png` (the 2011-2012 PM10 jump, smoothed PM10 measures and an IV with last year's PM10, with sources).
- `model_v1_energy_efficiency.py` creates `model_v1_energy_efficiency.md` and `figures/energy_efficiency.png` (are energy-efficient homes less affected by PM10? PM10 × average EPC rating of the homes of the same type in the postcode, efficiency groups A-C / D / E-G, robustness checks, data sources). Needs shapely, pyproj, scipy and pyarrow.
- `model_v1_green_space.py` creates `model_v1_green_space.md` and `figures/green_space.png` (does green space change the PM10 effect? PM10 × green share within 500 m, PM10 × local park within 400 m, green terciles, robustness checks, data sources). Needs scipy, shapely, pyproj and pyshp; uses `road_features.csv` from `model_v2_roads.py` for one check if it exists.
- `model_v2.py` creates `model_v2.md` and `figures/v2_interactions.png` (v2 model: log-quadratic distances, cubic borough pay, 3-year average PM10; interactions PM10 × house, PM10², PM10 × share aged 0-15/65+, PM10 × education). Needs `gla_lsoa_atlas/lsoa-data.csv`.
- `model_v2_roads.py` creates `model_v2_roads.md` and `figures/v2_roads.png` (v2 model with distance to the nearest motorway / A road and road noise: price effects, PM10 × near a main road, PM10 × noise, and whether main roads explain the larger PM10 discount for flats). Needs shapely, pyproj and pyshp.
- `model_v3.py` creates `model_v3.md` and `figures/v3_final.png` (the complete final model: inventory of everything we controlled for or tested, a control audit that decides which controls stay, the v3 model with 3-year average PM10, main roads, road noise and green space, and the interactions PM10 × house and PM10 × low green space, robustness checks). Needs `road_features.csv` and `green_features.csv` (built by `model_v2_roads.py` and `model_v1_green_space.py`, or on its first run if the road and green data are there).
- `air_pollution_maps.py` creates `air_pollution_maps.md` and the heat maps `figures/map_pm10.png`, `figures/map_nox.png` and `figures/map_pm10_by_year.png` (PM10 and NOx from our data, mean 2005-2014 per postcode, smoothed; with main roads, the Thames, borough lines and the borough means), each also with a white background (`figures/map_*_white.png`). Needs `london_boundaries/statistical-gis-boundaries-london.zip`; draws roads if `os_open_roads/oproad_essh_gb.zip` is there.
- `null_results.txt` is a running list of everything we tested that had no (robust) effect, with numbers and the source file.
