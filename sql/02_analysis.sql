-- Spatial analysis of the classified scene.
--
-- Every query works in EPSG:32631, the imagery's native UTM zone, so ST_Area
-- returns square metres directly with no geography cast. Areas are divided by
-- 1e6 for km2 at the point of presentation only.
--
-- Run with:
--   uv run python -m src.postprocessing.to_postgis --skip-load
--   docker exec -i sentinel2_postgis psql -U geouser -d landcover < sql/02_analysis.sql

-- Predicted land cover by class across the whole scene
SELECT
    class_id,
    class_name,
    count(*)                                    AS n_polygons,
    round((sum(ST_Area(geometry)) / 1e6)::numeric, 2) AS area_km2,
    round((100 * sum(ST_Area(geometry))
           / sum(sum(ST_Area(geometry))) OVER ())::numeric, 1) AS pct_of_scene
FROM landcover_polygons
GROUP BY class_id, class_name
ORDER BY area_km2 DESC;

-- Land cover per commune, only where the scene covers at least 90 percent of it
WITH per_commune AS (
    SELECT
        c.commune,
        c.lau2,
        c.covered_fraction,
        p.class_name,
        sum(ST_Area(ST_Intersection(p.geometry, c.geometry))) / 1e6 AS area_km2
    FROM commune_aoi c
    JOIN landcover_polygons p
      ON ST_Intersects(p.geometry, c.geometry)
    WHERE c.covered_fraction >= 0.90
    GROUP BY c.commune, c.lau2, c.covered_fraction, p.class_name
)
SELECT
    commune,
    lau2,
    round((100 * covered_fraction)::numeric, 0) AS pct_covered,
    round(sum(area_km2) FILTER (WHERE class_name LIKE '%forest%')::numeric, 2)    AS forest_km2,
    round(sum(area_km2) FILTER (WHERE class_name = 'Artificial surfaces')::numeric, 2) AS built_km2,
    round(sum(area_km2)::numeric, 2)            AS total_km2,
    round((100 * sum(area_km2) FILTER (WHERE class_name LIKE '%forest%')
           / NULLIF(sum(area_km2), 0))::numeric, 1) AS forest_pct
FROM per_commune
GROUP BY commune, lau2, covered_fraction
ORDER BY forest_pct DESC;

-- The five most built-up communes fully inside the scene
WITH per_commune AS (
    SELECT
        c.commune,
        sum(ST_Area(ST_Intersection(p.geometry, c.geometry))) FILTER
            (WHERE p.class_name = 'Artificial surfaces') / 1e6 AS built_km2,
        sum(ST_Area(ST_Intersection(p.geometry, c.geometry))) / 1e6 AS total_km2
    FROM commune_aoi c
    JOIN landcover_polygons p ON ST_Intersects(p.geometry, c.geometry)
    WHERE c.covered_fraction >= 0.90
    GROUP BY c.commune
)
SELECT
    commune,
    round(built_km2::numeric, 2)                                     AS built_km2,
    round((100 * built_km2 / NULLIF(total_km2, 0))::numeric, 1)      AS built_pct
FROM per_commune
ORDER BY built_pct DESC NULLS LAST
LIMIT 5;

-- Predicted area versus CORINE reference area, per class
WITH pred AS (
    SELECT p.class_name, sum(ST_Area(ST_Intersection(p.geometry, a.geometry))) / 1e6 AS km2
    FROM landcover_polygons p JOIN aoi_extent a ON ST_Intersects(p.geometry, a.geometry)
    GROUP BY p.class_name
), ref AS (
    SELECT c.class_name, sum(ST_Area(ST_Intersection(c.geometry, a.geometry))) / 1e6 AS km2
    FROM corine_polygons c JOIN aoi_extent a ON ST_Intersects(c.geometry, a.geometry)
    WHERE c.class_name IS NOT NULL
    GROUP BY c.class_name
)
SELECT
    coalesce(pred.class_name, ref.class_name)             AS class_name,
    round(coalesce(ref.km2, 0)::numeric, 2)               AS corine_km2,
    round(coalesce(pred.km2, 0)::numeric, 2)              AS predicted_km2,
    round((coalesce(pred.km2, 0) - coalesce(ref.km2, 0))::numeric, 2) AS difference_km2,
    round((100 * (coalesce(pred.km2, 0) - coalesce(ref.km2, 0))
           / NULLIF(ref.km2, 0))::numeric, 1)             AS pct_difference
FROM pred FULL OUTER JOIN ref ON pred.class_name = ref.class_name
ORDER BY abs(coalesce(pred.km2, 0) - coalesce(ref.km2, 0)) DESC;

-- Built-up land the model sees where CORINE 2018 does not, by commune
-- Candidate post-2018 development, but mostly CORINE generalisation: the 25 ha
-- minimum mapping unit cannot represent a new housing estate or a widened road,
-- so genuine small-scale sealing is invisible to CORINE by construction.
WITH new_built AS (
    SELECT
        c.commune,
        ST_Area(ST_Intersection(
            p.geometry,
            ST_Intersection(c.geometry, r.geometry)
        )) AS m2
    FROM landcover_polygons p
    JOIN commune_aoi c    ON ST_Intersects(p.geometry, c.geometry)
    JOIN corine_polygons r ON ST_Intersects(p.geometry, r.geometry)
                          AND ST_Intersects(r.geometry, c.geometry)
    WHERE p.class_name = 'Artificial surfaces'
      AND r.class_name IS DISTINCT FROM 'Artificial surfaces'
      AND c.covered_fraction >= 0.90
)
SELECT commune, round((sum(m2) / 1e6)::numeric, 3) AS built_outside_corine_km2
FROM new_built
GROUP BY commune
ORDER BY built_outside_corine_km2 DESC
LIMIT 10;

-- The ten largest contiguous predicted forest patches
-- Compactness is measured on a simplified outline, not the raw one. A polygon
-- traced from a raster has a staircase perimeter that follows pixel corners, so
-- its raw Polsby-Popper score measures the 10 m grid rather than the shape of
-- the wood -- every patch scores ~0.002 regardless of how round it is.
-- Simplifying to 30 m (three pixels) removes the staircase and leaves the
-- geometry an ecologist would recognise.
WITH patches AS (
    SELECT
        class_name,
        ST_Area(geometry)                                       AS area_m2,
        ST_Perimeter(geometry)                                  AS raw_perimeter_m,
        ST_SimplifyPreserveTopology(geometry, 30)               AS smooth_geom
    FROM landcover_polygons
    WHERE class_name LIKE '%forest%'
)
SELECT
    row_number() OVER (ORDER BY area_m2 DESC)     AS rank,
    class_name,
    round((area_m2 / 1e6)::numeric, 2)            AS area_km2,
    round(raw_perimeter_m::numeric, 0)            AS raw_perimeter_m,
    round(ST_Perimeter(smooth_geom)::numeric, 0)  AS simplified_perimeter_m,
    round((4 * pi() * ST_Area(smooth_geom)
           / NULLIF(power(ST_Perimeter(smooth_geom), 2), 0))::numeric, 3) AS compactness
FROM patches
ORDER BY area_m2 DESC
LIMIT 10;
