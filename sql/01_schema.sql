-- Schema helpers for the land-cover database.
--
-- The tables themselves are created by src/postprocessing/to_postgis.py via
-- GeoPandas. This script adds the things GeoPandas will not: validity repair,
-- a materialised AOI-clipped commune view, and the indexes the analysis
-- queries depend on.
--
-- Run with:
--   docker exec -i sentinel2_postgis psql -U geouser -d landcover < sql/01_schema.sql

CREATE EXTENSION IF NOT EXISTS postgis;

-- Polygons that come out of a raster tracer are usually valid, but CORINE's
-- photo-interpreted geometry sometimes has self-touching rings. An invalid
-- geometry makes ST_Intersection raise instead of returning a result, which
-- would abort a whole analysis query rather than degrade it.
UPDATE landcover_polygons SET geometry = ST_MakeValid(geometry)
 WHERE NOT ST_IsValid(geometry);
UPDATE corine_polygons    SET geometry = ST_MakeValid(geometry)
 WHERE NOT ST_IsValid(geometry);
UPDATE communes           SET geometry = ST_MakeValid(geometry)
 WHERE NOT ST_IsValid(geometry);

-- Communes clipped to the imaged area, with the fraction of each commune the
-- scene actually covers.
--
-- This view is the reason the analysis is trustworthy. A commune lying half
-- outside the 21 x 21 km scene has half its true area classified, and a naive
-- "forest area per commune" query would report it as half as forested as it is
-- -- with no hint in the output that anything was wrong. Carrying
-- `covered_fraction` alongside every area makes that visible, and lets queries
-- filter to communes the scene actually covers.
DROP MATERIALIZED VIEW IF EXISTS commune_aoi CASCADE;
CREATE MATERIALIZED VIEW commune_aoi AS
SELECT
    c.lau2,
    c.commune,
    c.canton,
    c.district,
    ST_Area(c.geometry) / 1e6                        AS commune_area_km2,
    ST_Area(ST_Intersection(c.geometry, a.geometry)) / 1e6 AS covered_area_km2,
    ST_Area(ST_Intersection(c.geometry, a.geometry))
        / NULLIF(ST_Area(c.geometry), 0)             AS covered_fraction,
    ST_Intersection(c.geometry, a.geometry)          AS geometry
FROM communes c
JOIN aoi_extent a ON ST_Intersects(c.geometry, a.geometry)
WHERE ST_Area(ST_Intersection(c.geometry, a.geometry)) > 0;

CREATE INDEX IF NOT EXISTS commune_aoi_geom_idx ON commune_aoi USING GIST (geometry);
CREATE INDEX IF NOT EXISTS landcover_class_idx  ON landcover_polygons (class_id);
CREATE INDEX IF NOT EXISTS corine_class_idx     ON corine_polygons (class_id);

ANALYZE landcover_polygons;
ANALYZE corine_polygons;
ANALYZE commune_aoi;
