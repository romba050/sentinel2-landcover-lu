# QGIS layers

The `.qml` style files are generated from `src.labels.corine.CLASSES`, the same
definition the label rasters, the PostGIS tables and the report figures use:

```bash
uv run python -m src.postprocessing.qgis_styles
```

Regenerate them after changing the class scheme; do not edit them by hand.

## Loading the layers

**Rasters** — *Layer ▸ Add Layer ▸ Add Raster Layer*:

| File | Contents |
|---|---|
| `data/processed/*_stack.tif` | 15-band reflectance + indices stack (EPSG:32631) |
| `data/processed/*_labels.tif` | CORINE 2018 rasterised to the 10 m grid |
| `data/processed/*_label_boundary_dist.tif` | metres to the nearest class boundary |
| `data/results/*_rf_prediction.tif` | Random Forest classification |

Apply `landcover_raster.qml` to the label and prediction rasters via *Layer
Properties ▸ Symbology ▸ Style ▸ Load Style*. They share a palette on purpose,
so flipping between them shows real disagreement rather than a colour change.

For the reflectance stack, set *Render type: Multiband color* with
R=3 (B04), G=2 (B03), B=1 (B02) for true colour, or R=4 (B08), G=3, B=2 for
false-colour infrared. Use *Cumulative count cut 2%–98%* — the raw min/max
stretch is ruined by a handful of specular roof pixels above 1.0 reflectance.

**PostGIS** — *Layer ▸ Add Layer ▸ Add PostGIS Layers ▸ New*:

| Field | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Database | `landcover` |
| Username | `geouser` |
| Password | `geopass` |

Start the database first with `docker compose -f docker/docker-compose.yml up -d`.
Available tables: `landcover_polygons`, `communes`, `corine_polygons`,
`aoi_extent`, and the `commune_aoi` view. Apply `landcover_polygons.qml` to
`landcover_polygons`.

Note these are throwaway local development credentials that are deliberately
committed so the stack runs with one command. Do not reuse this compose file
anywhere reachable from a network.

## Project CRS

Set the project CRS to **EPSG:32631** (WGS 84 / UTM zone 31N) to match the
imagery. QGIS will reproject the commune boundaries on the fly for display.

This is UTM zone **31**, not the 32 you would expect for Luxembourg. The country
straddles the 6°E zone boundary, and the AOI used here is only fully contained
by MGRS tile 31UGR. The CRS is read from the Sentinel-2 scene rather than
assumed — see `src/ingest/sentinel2.py`.

## A map layout worth exporting

1. Set the project CRS to EPSG:32631.
2. Load the true-colour stack, then the prediction raster above it.
3. *Project ▸ New Print Layout*.
4. Add two side-by-side maps — imagery and classification — locked to the same
   extent and scale so they are honestly comparable.
5. Add the commune boundaries as an outline-only overlay on both.
6. Add a scale bar (metric), a north arrow, and a legend generated from the
   `.qml` categories.
7. Credit the sources in the layout: *"Contains modified Copernicus Sentinel
   data 2024; CORINE Land Cover 2018 © European Environment Agency; commune
   boundaries © Administration du cadastre et de la topographie, data.public.lu"*.
8. Export to PNG at 300 dpi into `data/results/figures/`.

The layout step is the one part of this project that needs QGIS itself
installed; everything above it is reproducible from the command line.
