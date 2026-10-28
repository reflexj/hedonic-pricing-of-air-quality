"""
HE3027 Group Project: heat maps of air pollution in London (PM10 and NOx) from our own data.

The maps are styled like the London Atmospheric Emissions Inventory maps: a dark background outside London and a
smooth blue-to-red surface inside, with main roads, the Thames, borough lines and one circle per borough.

What the colour shows: our PM10 (and NOx) is the yearly mean at the monitoring site nearest to each home, as given in
the course data. For every postcode with a sale we take the mean over 2005-2014 (first per year, then over the years),
and smooth these postcode values into a surface with a Gaussian kernel. The map therefore shows the pollution our
models use, not a modelled concentration field: it cannot show the extra pollution right next to a busy road. The main
roads are drawn on top for orientation only.

Creates:
    figures/map_pm10.png           PM10, mean 2005-2014
    figures/map_nox.png            NOx, mean 2005-2014
    figures/map_pm10_by_year.png   PM10 in each year 2005-2014, one colour scale for all years
    figures/map_*_white.png        the same three maps with a white background (for print)
    air_pollution_maps.md          the maps with notes on the method and the sources

Run from the Code folder:
    python air_pollution_maps.py

Needs: pandas, numpy, matplotlib, scipy, shapely, pyproj, pyshp
Data:  data/land_registry_pollution.dta
       data/london_boundaries/statistical-gis-boundaries-london.zip   (keep zipped)
           Statistical GIS Boundary Files for London (London Datastore, Open Government Licence):
           https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london
       data/os_open_roads/oproad_essh_gb.zip   (as for model_v2_roads.py; optional, the maps work without roads)
"""

import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
import shapely
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))   # the model scripts live in models/

from model_v1 import CODE_DIR, DATA_DIR, LAST_YEAR, load_data

MD_FILE = CODE_DIR / "air_pollution_maps.md"
FIG_DIR = CODE_DIR / "figures"
BOUNDARY_ZIP = DATA_DIR / "london_boundaries" / "statistical-gis-boundaries-london.zip"
ROADS_ZIP = DATA_DIR / "os_open_roads" / "oproad_essh_gb.zip"
ROAD_TILES = ["TQ", "TL"]
ROAD_CLASSES = {"Motorway": 1.1, "A Road": 0.6, "B Road": 0.3}     # OS Open Roads `class`: line width

RES_M = 100          # grid cell of the surface (m)
SIGMA_M = 750        # Gaussian smoothing of the postcode values (m)
SIGMA_FILL_M = 3000  # wider smoothing that fills large gaps without postcodes (parks, reservoirs, airports)
MIN_WEIGHT = 3       # fewer postcodes than this within the near kernel: use the wide surface
CLOSE_M = 400        # closing the borough outlines across the Thames to get one London shape
MARGIN_M = 4000      # map extent around London

ROAD_IN = "#3a1208"        # roads inside London (dark lines on the colour surface)
THAMES = "#2f6fd6"
CMAP = "turbo"             # blue - cyan - green - yellow - red, as in the LAEI maps

THEMES = {  # every map is drawn twice: dark background (as the LAEI maps) and white background (for print)
    "dark": {"bg": "#17181a", "road_out": "#3d3f42", "text": "#e8e6e1", "subtle": "#a9a7a1", "border": "white",
             "outline": "white", "suffix": ""},
    "white": {"bg": "white", "road_out": "#d6d5d1", "text": "#1d1d1b", "subtle": "#5f5e5a", "border": "#3d3d3a",
              "outline": "#2b2b2b", "suffix": "_white"},
}

POLLUTANTS = {"pm10": ("PM10", "µg/m³"), "nox": ("NOx", "µg/m³")}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read_boroughs():
    """The 33 London boroughs (British National Grid), clipped at the Thames (mean high water)."""
    with zipfile.ZipFile(BOUNDARY_ZIP) as z:
        base = next(n[:-4] for n in z.namelist() if n.endswith("London_Borough_Excluding_MHW.shp"))
        r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), shx=io.BytesIO(z.read(base + ".shx")),
                             dbf=io.BytesIO(z.read(base + ".dbf")), encoding="latin-1")
        fields = [f[0] for f in r.fields[1:]]
        name_field = "NAME" if "NAME" in fields else fields[0]
        names, polys = [], []
        for sr in r.iterShapeRecords():
            names.append(sr.record[fields.index(name_field)])
            polys.append(shapely.make_valid(shapely.geometry.shape(sr.shape.__geo_interface__)))
    return pd.DataFrame({"name": names, "geometry": polys})


def london_shapes(boroughs):
    """London as one shape (with the Thames closed) and the Thames as the gap between the boroughs."""
    union = shapely.union_all(boroughs["geometry"].to_numpy())
    closed = union.buffer(CLOSE_M).buffer(-CLOSE_M)
    closed = shapely.Polygon(max(getattr(closed, "geoms", [closed]), key=lambda g: g.area).exterior)
    thames = closed.difference(union).buffer(-5).buffer(5)       # drop slivers along the borough edges
    parts = [g for g in getattr(thames, "geoms", [thames]) if g.area > 200_000]   # the river, not small gaps
    return closed, shapely.union_all(parts) if parts else shapely.Polygon()


def read_roads():
    """Motorways, A and B roads from OS Open Roads; empty if the file is not there."""
    if not ROADS_ZIP.exists():
        return [], []
    lines, width = [], []
    with zipfile.ZipFile(ROADS_ZIP) as z:
        for tile in ROAD_TILES:
            base = next(n[:-4] for n in z.namelist() if n.endswith(f"{tile}_RoadLink.shp"))
            r = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), shx=io.BytesIO(z.read(base + ".shx")),
                                 dbf=io.BytesIO(z.read(base + ".dbf")))
            i_class = [f[0] for f in r.fields[1:]].index("class")
            for i, rec in enumerate(r.iterRecords()):
                w = ROAD_CLASSES.get(rec[i_class])
                if w:
                    lines.append(np.asarray(r.shape(i).points))
                    width.append(w)
    return lines, np.array(width)


def postcode_means(df):
    """Mean pollution per postcode over the years: first per postcode and year, then over the years."""
    e, n = Transformer.from_crs(4326, 27700, always_xy=True).transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    d = df.assign(e=e, n=n)
    py = d.groupby(["postcode", "year"]).agg(pm10=("pm10", "mean"), nox=("nox", "mean"), e=("e", "first"),
                                             n=("n", "first")).reset_index()
    pc = py.groupby("postcode").agg(pm10=("pm10", "mean"), nox=("nox", "mean"), e=("e", "first"), n=("n", "first"))
    return pc, py


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------
class Grid:
    def __init__(self, shape_bounds):
        a, b, c, d = shape_bounds
        self.e0, self.n0 = a - MARGIN_M, b - MARGIN_M
        self.e1, self.n1 = c + MARGIN_M, d + MARGIN_M
        self.nx = int((self.e1 - self.e0) / RES_M)
        self.ny = int((self.n1 - self.n0) / RES_M)

    @property
    def extent(self):
        return (self.e0, self.e0 + self.nx * RES_M, self.n0, self.n0 + self.ny * RES_M)

    def mask(self, poly):
        X, Y = np.meshgrid(self.e0 + (np.arange(self.nx) + 0.5) * RES_M, self.n0 + (np.arange(self.ny) + 0.5) * RES_M)
        shapely.prepare(poly)
        return shapely.contains_xy(poly, X, Y)

    def surface(self, e, n, v, inside):
        """Gaussian-weighted local mean of the postcode values (each postcode counts once), NaN outside London."""
        i = np.clip(((e - self.e0) / RES_M).astype(int), 0, self.nx - 1)
        j = np.clip(((n - self.n0) / RES_M).astype(int), 0, self.ny - 1)
        s = np.zeros((self.ny, self.nx))
        w = np.zeros((self.ny, self.nx))
        np.add.at(s, (j, i), v)
        np.add.at(w, (j, i), 1.0)
        out = {}
        for name, sig in [("near", SIGMA_M), ("wide", SIGMA_FILL_M)]:
            ws = gaussian_filter(w, sig / RES_M)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[name] = gaussian_filter(s, sig / RES_M) / ws
            out[name + "_w"] = ws * (2 * np.pi * (sig / RES_M) ** 2)   # ~ number of postcodes within the kernel
        z = np.where(out["near_w"] >= MIN_WEIGHT, out["near"], out["wide"])
        return np.where(inside, z, np.nan)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def poly_path(geom):
    """matplotlib Path of a (Multi)Polygon, holes included."""
    verts, codes = [], []
    for p in getattr(geom, "geoms", [geom]):
        if p.is_empty or p.geom_type != "Polygon":
            continue
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes.append([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
    if not verts:
        return None
    return Path(np.concatenate(verts), np.concatenate(codes))


def draw_base(ax, t, grid, london, thames, boroughs, roads, z, vmin, vmax, road_scale=1.0, borough_lw=0.5):
    ax.set_facecolor(t["bg"])
    lines, width = roads
    if len(lines):
        start = np.array([ln[0] for ln in lines])
        shapely.prepare(london)
        inside = shapely.contains_xy(london, start[:, 0], start[:, 1])
        out_idx = np.where(~inside)[0]
        ax.add_collection(LineCollection([lines[k] for k in out_idx], colors=t["road_out"],
                                         linewidths=width[out_idx] * road_scale, zorder=1))
    im = ax.imshow(z, extent=grid.extent, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax, interpolation="bilinear",
                   zorder=2)
    if len(lines):
        in_idx = np.where(inside)[0]
        ax.add_collection(LineCollection([lines[k] for k in in_idx], colors=ROAD_IN, alpha=0.4,
                                         linewidths=width[in_idx] * road_scale, zorder=3))
    for g in boroughs["geometry"]:
        p = poly_path(g)
        if p is not None:
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor=t["border"], linewidth=borough_lw, alpha=0.35,
                                   zorder=4))
    tp = poly_path(thames)
    if tp is not None:
        ax.add_patch(PathPatch(tp, facecolor=THAMES, edgecolor=THAMES, linewidth=1.2, zorder=5))
    lp = poly_path(london)
    ax.add_patch(PathPatch(lp, facecolor="none", edgecolor=t["outline"], linewidth=1.0, alpha=0.8, zorder=6))
    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def borough_circles(ax, boroughs, pc, col, london_mean):
    """One circle per borough with its mean (over postcodes): orange above the London mean, green below."""
    pts = shapely.points(pc["e"].to_numpy(), pc["n"].to_numpy())
    for _, b in boroughs.iterrows():
        shapely.prepare(b["geometry"])
        sel = shapely.contains(b["geometry"], pts)
        if sel.sum() == 0:
            continue
        m = pc.loc[sel, col].mean()
        c = b["geometry"].representative_point()
        color = "#ef7d22" if m > london_mean else "#4c9a52"
        ax.scatter(c.x, c.y, s=260, color=color, edgecolor="white", linewidth=0.8, zorder=7)
        ax.annotate(f"{m:.0f}", (c.x, c.y), ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold", zorder=8)


def footer(fig, t, text, y=0.012):
    fig.text(0.015, y, text, fontsize=6.5, color=t["subtle"], ha="left", va="bottom", wrap=True)


SOURCE = ("Data: course data set (PM10 / NOx at the nearest monitoring site, Land Registry sales). Contains National "
          "Statistics data © Crown copyright and database right 2015. Contains OS data © Crown copyright and "
          "database right (boundaries 2015, OS Open Roads current release).")


def map_one(t, col, grid, london, thames, boroughs, roads, pc, inside):
    name, unit = POLLUTANTS[col]
    z = grid.surface(pc["e"].to_numpy(), pc["n"].to_numpy(), pc[col].to_numpy(), inside)
    vmin, vmax = np.nanpercentile(z, [1, 99])
    # figure shaped like the map (London is about 1.25 times as wide as tall), so there are no empty side bands
    fig = plt.figure(figsize=(8.4, 7.9), facecolor=t["bg"])
    ax = fig.add_axes((0.01, 0.16, 0.98, 0.73))
    im = draw_base(ax, t, grid, london, thames, boroughs, roads, z, vmin, vmax)
    mean = pc[col].mean()
    borough_circles(ax, boroughs, pc, col, mean)
    fig.text(0.03, 0.955, f"Air pollution in London: {name}", fontsize=15, color=t["text"], fontweight="bold")
    fig.text(0.03, 0.925, f"Mean 2005-{LAST_YEAR} at the nearest monitoring site, smoothed over "
             f"{len(pc):,} postcodes with sales", fontsize=8.5, color=t["subtle"])
    cax = fig.add_axes((0.05, 0.11, 0.50, 0.017))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{name} ({unit})", color=t["text"], fontsize=8.5)
    cb.ax.tick_params(colors=t["text"], labelsize=7.5)
    cb.outline.set_visible(False)
    dots = [plt.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=c, markeredgecolor="white",
                       markeredgewidth=0.8) for c in ["#ef7d22", "#4c9a52"]]
    leg = fig.legend(dots, [f"above the London mean ({mean:.1f})", "below the London mean"],
                     title=f"Circles: borough mean ({unit})", loc="lower right", bbox_to_anchor=(0.97, 0.075),
                     frameon=False, fontsize=7.5, title_fontsize=7.5, labelcolor=t["text"], alignment="left")
    leg.get_title().set_color(t["subtle"])
    footer(fig, t, SOURCE)
    path = FIG_DIR / f"map_{col}{t['suffix']}.png"
    fig.savefig(path, dpi=220, facecolor=t["bg"])
    plt.close(fig)
    return path, z


def map_by_year(t, grid, london, thames, boroughs, py, inside):
    years = sorted(py["year"].unique())
    zs = {y: grid.surface(*(py.loc[py["year"] == y, c].to_numpy() for c in ["e", "n", "pm10"]), inside)
          for y in years}
    vmin, vmax = np.nanpercentile(np.concatenate([z[np.isfinite(z)] for z in zs.values()]), [1, 99])
    ncol = 5
    nrow = int(np.ceil(len(years) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.55 * nrow + 1.2), facecolor=t["bg"])
    for ax, y in zip(axes.flat, years):
        im = draw_base(ax, t, grid, london, thames, boroughs, ([], []), zs[y], vmin, vmax, borough_lw=0.3)
        mean_y = py.loc[py["year"] == y, "pm10"].mean()
        ax.set_title(f"{y}  (mean {mean_y:.1f})", color=t["text"], fontsize=9.5, pad=3)
    for ax in list(axes.flat)[len(years):]:
        ax.set_visible(False)
    fig.suptitle("PM10 in London, year by year (same colour scale in every panel)", color=t["text"], fontsize=13,
                 x=0.015, ha="left", y=0.985, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, wspace=0.03, hspace=0.18)
    cax = fig.add_axes((0.35, 0.115, 0.30, 0.02))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("PM10 (µg/m³), mean over the postcodes with sales that year", color=t["text"], fontsize=8.5)
    cb.ax.tick_params(colors=t["text"], labelsize=7.5)
    cb.outline.set_visible(False)
    footer(fig, t, SOURCE)
    path = FIG_DIR / f"map_pm10_by_year{t['suffix']}.png"
    fig.savefig(path, dpi=200, facecolor=t["bg"])
    plt.close(fig)
    return path, {y: float(py.loc[py["year"] == y, "pm10"].mean()) for y in years}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(exist_ok=True)
    df, _, _ = load_data()
    pc, py = postcode_means(df)
    boroughs = read_boroughs()
    london, thames = london_shapes(boroughs)
    grid = Grid(london.bounds)
    inside = grid.mask(london)
    roads = read_roads()
    print(f"{len(pc):,} postcodes, {len(boroughs)} boroughs, {len(roads[0]):,} road links")

    stats = {}
    for t in THEMES.values():
        for col in POLLUTANTS:
            path, z = map_one(t, col, grid, london, thames, boroughs, roads, pc, inside)
            stats[col] = (np.nanpercentile(z, 5), np.nanpercentile(z, 95), pc[col].mean())
            print(f"Wrote {path}")
        year_path, year_means = map_by_year(t, grid, london, thames, boroughs, py, inside)
        print(f"Wrote {year_path}")

    year_md = " | ".join(f"{y}: {m:.1f}" for y, m in year_means.items())
    md = f"""# Air Pollution in London: Heat Maps

Generated by `air_pollution_maps.py` from our own data.

![PM10 in London](figures/map_pm10.png)

![NOx in London](figures/map_nox.png)

![PM10 year by year](figures/map_pm10_by_year.png)

Every map also exists with a white background, for print and for the report: `figures/map_pm10_white.png`,
`figures/map_nox_white.png` and `figures/map_pm10_by_year_white.png`.

## What the maps show

- **Colour:** the yearly mean of PM10 (or NOx) at the monitoring site nearest to each home, as in the course data and
  in all our models. For each of the {len(pc):,} postcodes with a sale we take the mean over 2005-{LAST_YEAR} (first
  per year, then over the years, so busy years do not count more), and smooth these values with a Gaussian kernel
  (σ = {SIGMA_M} m on a {RES_M} m grid; where there are almost no postcodes, such as large parks or reservoirs,
  σ = {SIGMA_FILL_M / 1000:.0f} km). Every postcode counts once, so dense areas do not dominate.
- **Circles:** the mean over the postcodes of each borough. Orange: above the London mean, green: below.
- **Roads, Thames and borough lines** are drawn for orientation. The roads are not part of the pollution values.

| Pollutant | London mean (postcodes) | 5th percentile of the map | 95th percentile of the map |
|---|---|---|---|
| PM10 (µg/m³) | {stats['pm10'][2]:.1f} | {stats['pm10'][0]:.1f} | {stats['pm10'][1]:.1f} |
| NOx (µg/m³) | {stats['nox'][2]:.1f} | {stats['nox'][0]:.1f} | {stats['nox'][1]:.1f} |

Mean PM10 over the postcodes with sales, by year: {year_md}.

## What the maps do not show

- **No road-level detail.** Maps such as the London Atmospheric Emissions Inventory (LAEI) model the concentration
  for every 20 m and show the extra pollution along busy roads. Our values come from a few dozen monitoring sites:
  each home gets the value of its nearest site, so the true pattern in our data is a set of areas around the sites,
  which the smoothing blurs into a continuous surface. Local hot spots along main roads are not in the data.
- **Only where homes were sold.** The surface is built from postcodes with at least one sale in 2005-{LAST_YEAR}.
  Large parks, the airports and industrial areas get the values of the homes around them.

## Sources

- PM10, NOx and the sales: course data set `land_registry_pollution.dta`.
- Borough boundaries (2011, clipped at mean high water; the Thames is the gap between them):
  [Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london),
  London Datastore, Open Government Licence. Contains National Statistics data © Crown copyright and database right
  2015. Contains Ordnance Survey data © Crown copyright and database right 2015.
- Roads: [OS Open Roads](https://osdatahub.os.uk/downloads/open/OpenRoads) (motorways, A and B roads), current
  release, Open Government Licence. Contains OS data © Crown copyright and database right.
"""
    MD_FILE.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_FILE}")


if __name__ == "__main__":
    main()
