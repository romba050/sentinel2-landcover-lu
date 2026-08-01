"""Vectorise the classified raster and load it into PostGIS with commune boundaries.

Why leave the raster at all
---------------------------
A raster answers "what is at this point". The questions a land-monitoring user
actually asks are relational -- "how much forest does each commune have", "which
communes are most built-up", "how much predicted urban area sits outside what
CORINE calls urban". Those are joins between geometries, which is what a spatial
database is for.

What gets loaded
----------------
``landcover_polygons``  the RF prediction, vectorised and dissolved by class
``communes``            Luxembourg's 100 communes (LAU2), from data.public.lu
``corine_polygons``     the CLC2018 reference, for like-for-like comparison

Everything is stored in the imagery's native UTM CRS (EPSG:32631) rather than
WGS84, so that ``ST_Area`` returns square metres directly and no per-query
geography cast or reprojection is needed.

    docker compose -f docker/docker-compose.yml up -d
    uv run python -m src.postprocessing.to_postgis
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.features import shapes as raster_shapes
from shapely.geometry import box, shape
from sqlalchemy import create_engine, text

from src.labels.corine import CLC_TO_CLASS

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "data" / "results"
RAW = PROJECT_ROOT / "data" / "raw"
SQL_DIR = PROJECT_ROOT / "sql"

DEFAULT_DSN = os.environ.get(
    "LANDCOVER_DSN", "postgresql+psycopg2://geouser:geopass@localhost:5432/landcover"
)

#: Drop vectorised specks below this area. A single 10 m pixel becomes a 100 m2
#: polygon; keeping every one of them yields hundreds of thousands of features
#: that slow every join down and carry no information a user would act on.
MIN_POLYGON_AREA_M2 = 2_000.0


def vectorise_prediction(
    raster_path: Path, class_names: dict[int, str], min_area_m2: float
) -> gpd.GeoDataFrame:
    """Polygonise the class raster, dropping specks below ``min_area_m2``."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        crs, transform = src.crs, src.transform

    records = []
    dropped_n, dropped_area = 0, 0.0
    # mask=data > 0 keeps nodata out of the polygonisation entirely.
    for geom, value in raster_shapes(data, mask=data > 0, transform=transform):
        polygon = shape(geom)
        if polygon.area < min_area_m2:
            dropped_n += 1
            dropped_area += polygon.area
            continue
        cls = int(value)
        records.append({"class_id": cls, "class_name": class_names.get(cls, "?"),
                        "geometry": polygon})

    gdf = gpd.GeoDataFrame(records, crs=crs)
    total = gdf.area.sum() + dropped_area
    # Report what the filter removed rather than letting the area quietly not
    # add up: a reader comparing the table totals to the AOI size deserves to
    # know that the difference is speckle, not missing classification.
    log.info(
        "Vectorised %s -> %d polygons >= %.0f m2; dropped %d specks "
        "totalling %.1f km2 (%.1f%% of the classified area)",
        raster_path.name, len(gdf), min_area_m2, dropped_n,
        dropped_area / 1e6, 100 * dropped_area / total,
    )
    gdf.attrs["dropped_speck_area_km2"] = dropped_area / 1e6
    gdf.attrs["dropped_speck_count"] = dropped_n
    return gdf


def load_communes(path: Path, crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path).to_crs(crs)
    gdf = gdf.rename(columns=str.lower)[["commune", "canton", "district", "lau2", "geometry"]]
    log.info("Communes: %d features reprojected to %s", len(gdf), crs)
    return gdf


def write_table(gdf: gpd.GeoDataFrame, engine, name: str):
    gdf.to_postgis(name, engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name}_geom_idx "
                          f"ON {name} USING GIST (geometry)"))
        conn.execute(text(f"ANALYZE {name}"))
    log.info("loaded %-20s %6d rows", name, len(gdf))


def run(dsn: str = DEFAULT_DSN, min_area_m2: float = MIN_POLYGON_AREA_M2) -> dict:
    rf = json.loads((RESULTS / "latest_rf.json").read_text())
    labels_meta = json.loads((PROCESSED / "latest_labels.json").read_text())
    kept = [c for c in labels_meta["class_scheme"]["classes"] if c["kept"]]
    class_names = {c["class_id"]: c["name"] for c in kept}

    prediction_path = RESULTS / rf["files"]["prediction"]
    prediction = vectorise_prediction(prediction_path, class_names, min_area_m2)
    crs = prediction.crs

    communes = load_communes(RAW / "communes4326.geojson", crs)
    corine = gpd.read_file(RAW / Path(labels_meta["files"]["vector"]).name).to_crs(crs)
    corine = corine.rename(columns={"Code_18": "clc_code", "Area_Ha": "area_ha"})
    corine["class_id"] = corine["clc_code"].astype(int).map(CLC_TO_CLASS)
    corine["class_name"] = corine["class_id"].map(class_names)
    corine = corine[["clc_code", "class_id", "class_name", "area_ha", "geometry"]]

    # The AOI footprint is a first-class table. Without it, per-commune areas are
    # meaningless: a commune only half-covered by the scene would look half as
    # forested as it is, and nothing in the query would reveal that.
    with rasterio.open(prediction_path) as src:
        aoi = gpd.GeoDataFrame(
            {"name": ["sentinel2_aoi"]}, geometry=[box(*src.bounds)], crs=crs
        )

    engine = create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        # Views built by 01_schema.sql depend on these tables, and GeoPandas'
        # if_exists="replace" issues a bare DROP TABLE that Postgres refuses
        # while a dependent view exists. Dropping them first makes re-running
        # the loader idempotent instead of a one-shot.
        conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS commune_aoi CASCADE"))

    write_table(prediction, engine, "landcover_polygons")
    write_table(communes, engine, "communes")
    write_table(corine, engine, "corine_polygons")
    write_table(aoi, engine, "aoi_extent")

    with engine.begin() as conn:
        srid = conn.execute(
            text("SELECT ST_SRID(geometry) FROM landcover_polygons LIMIT 1")
        ).scalar()
        counts = {
            t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar()
            for t in ("landcover_polygons", "communes", "corine_polygons")
        }
    log.info("SRID in database: %s", srid)
    return {"srid": srid, "row_counts": counts,
            "min_polygon_area_m2": min_area_m2,
            "predicted_area_km2": float(prediction.area.sum() / 1e6),
            "dropped_speck_area_km2": prediction.attrs["dropped_speck_area_km2"],
            "dropped_speck_count": prediction.attrs["dropped_speck_count"]}


def _split_statements(sql: str) -> list[str]:
    return [s.strip() for s in sql.split(";") if s.strip()]


def _first_code_line(stmt: str) -> int:
    for i, line in enumerate(stmt.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            return i
    return -1


def statement_title(stmt: str) -> str:
    """The heading for a statement: the first line of the comment block above it.

    Walking *backwards* from the first line of code, rather than taking the
    first comment in the chunk, keeps a file-level header from being mistaken
    for the first query's title.
    """
    lines = stmt.splitlines()
    start = _first_code_line(stmt)
    if start <= 0:
        return "query"
    block: list[str] = []
    for line in reversed(lines[:start]):
        stripped = line.strip()
        if not stripped.startswith("--"):
            break
        block.append(stripped.lstrip("-").strip())
    return block[-1] if block else "query"


def is_query(stmt: str) -> bool:
    """True if the statement returns rows (comments stripped before deciding)."""
    start = _first_code_line(stmt)
    if start < 0:
        return False
    keyword = stmt.splitlines()[start].strip().split()[0].upper()
    return keyword in {"SELECT", "WITH", "TABLE", "VALUES"}


def execute_script(dsn: str, sql_path: Path) -> str:
    """Run a .sql file, pretty-printing any result sets it produces."""
    import pandas as pd

    engine = create_engine(dsn)
    out: list[str] = []
    with engine.begin() as conn:
        for stmt in _split_statements(sql_path.read_text()):
            if not is_query(stmt):
                conn.execute(text(stmt))
                continue
            df = pd.read_sql_query(text(stmt), conn)
            out.append(f"\n--- {statement_title(stmt)} ---\n"
                       f"{df.to_string(index=False, max_rows=30)}")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--min-area", type=float, default=MIN_POLYGON_AREA_M2)
    p.add_argument("--schema", type=Path, default=SQL_DIR / "01_schema.sql")
    p.add_argument("--analysis", type=Path, default=SQL_DIR / "02_analysis.sql")
    p.add_argument("--skip-load", action="store_true", help="Only run the SQL.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    if not args.skip_load:
        info = run(args.dsn, args.min_area)
        print(f"\nLoaded into PostGIS (SRID {info['srid']}):")
        for table, n in info["row_counts"].items():
            print(f"  {table:<22} {n:>7,} rows")
        print(f"  predicted area          {info['predicted_area_km2']:>7.1f} km2")
        print(f"  speckle removed         {info['dropped_speck_area_km2']:>7.1f} km2 in "
              f"{info['dropped_speck_count']:,} polygons")

    if args.schema.exists():
        execute_script(args.dsn, args.schema)
        log.info("applied %s", args.schema.name)

    if args.analysis.exists():
        report = execute_script(args.dsn, args.analysis)
        print(report)
        out = RESULTS / "postgis_analysis.txt"
        out.write_text(report)
        print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
