-- Forest share per commune, WITH geometry -- for loading as a QGIS layer.
--
-- The per-commune query in 02_analysis.sql was written for a text report and
-- returns no geometry column, so DB Manager cannot put its result on the map.
-- This variant joins the commune geometry back on, which is what QGIS needs.
--
-- In QGIS: Database > DB Manager > PostgreSQL > landcover > SQL window,
-- paste this, Execute, tick "Load as new layer",
-- Geometry column = geometry, Column with unique values = lau2, Load.
-- Then style: Symbology > Graduated, value forest_pct, Greens ramp, ~5 classes.

WITH per_commune AS (
    SELECT
        c.lau2,
        c.commune,
        p.class_name,
        sum(ST_Area(ST_Intersection(p.geometry, c.geometry))) / 1e6 AS area_km2
    FROM commune_aoi c
    JOIN landcover_polygons p ON ST_Intersects(p.geometry, c.geometry)
    WHERE c.covered_fraction >= 0.90
    GROUP BY c.lau2, c.commune, p.class_name
)
SELECT
    pc.lau2,
    pc.commune,
    round((100 * sum(pc.area_km2) FILTER (WHERE pc.class_name LIKE '%forest%')
           / NULLIF(sum(pc.area_km2), 0))::numeric, 1) AS forest_pct,
    round((100 * sum(pc.area_km2) FILTER (WHERE pc.class_name = 'Artificial surfaces')
           / NULLIF(sum(pc.area_km2), 0))::numeric, 1) AS built_pct,
    round(sum(pc.area_km2)::numeric, 2) AS classified_km2,
    ca.geometry
FROM per_commune pc
JOIN commune_aoi ca USING (lau2)
GROUP BY pc.lau2, pc.commune, ca.geometry;
