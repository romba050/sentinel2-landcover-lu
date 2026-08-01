# Sentinel-2 land-cover classification over Luxembourg

An end-to-end land-cover pipeline for a 21 × 21 km window around Luxembourg
City: Sentinel-2 L2A imagery from a STAC API, CORINE Land Cover labels from the
European Environment Agency, a Random Forest evaluated with **spatially-blocked
validation**, and the result loaded into PostGIS for per-commune analysis.

![Sentinel-2 quicklook](data/results/figures/luxembourg-city_31UGR_2024-08-24_quicklook.jpg)

> **What this is.** A self-directed learning project, built in about a week in
> late July / August 2026 to transfer my MSc work on pixel-wise semantic
> segmentation (mean-field networks + CRF, retinal vessel segmentation) onto
> satellite imagery. It is not production work and does not represent prior
> professional Earth-observation experience. It was built with Claude Code
> (agentic development); every design decision in it is one I can defend, and
> the things that went wrong are documented below rather than quietly fixed.

---

## Headline results

Random Forest, 15 per-pixel spectral features, no spatial context, evaluated on
**1.35 million held-out pixels** in spatial blocks the model never saw:

| Metric | Spatially-blocked | Random pixel split |
|---|---|---|
| Overall accuracy | **0.624** | 0.647 |
| Cohen's κ | **0.523** | 0.550 |
| Macro F1 | **0.415** | 0.446 |

The blocked figure is the one to quote. Three findings matter more than the
headline number.

### 1. A random pixel split measures interpolation, not generalisation

Land cover is spatially autocorrelated, and CORINE polygons are at least 25 ha,
so adjacent pixels usually share both their spectrum *and* their label. Split
those at random and nearly every test pixel has a near-duplicate in training.

Sweeping the training set from 25 k to 1.6 M pixels shows exactly that:

![Leakage vs training density](data/results/figures/fig3_leakage_vs_training_density.png)

| Training pixels | Blocked | Random | Gap |
|---:|---:|---:|---:|
| 25,000 | 0.614 | 0.625 | +0.011 |
| 100,000 | 0.621 | 0.638 | +0.017 |
| 400,000 | 0.625 | 0.648 | +0.023 |
| 1,600,000 | 0.626 | 0.661 | +0.035 |

64× more training data moves the blocked score by 0.012 and the random score by
0.036. The extra pixels teach the model almost nothing new about the landscape;
under a random split they mostly serve as lookups for their own neighbours. Any
single number for "how much leakage costs" would have been an artefact of the
sample size chosen to measure it.

### 2. Most of the residual error is label error, not model error

CORINE is photo-interpreted at 1:100 000 with a 25 ha minimum mapping unit.
**49.5%** of labelled pixels here lie within 100 m of a class boundary — the
scale at which those boundaries are actually accurate.

![Accuracy vs boundary distance](data/results/figures/fig2_accuracy_vs_boundary_distance.png)

| Distance to nearest CORINE boundary | Accuracy |
|---|---|
| 0–20 m | 0.437 |
| 20–50 m | 0.559 |
| 50–100 m | 0.604 |
| 100–200 m | 0.633 |
| 200–400 m | 0.700 |
| > 400 m | **0.813** |

Deep inside a polygon, where the label is trustworthy, the same model scores
0.81. The 0.62 headline is substantially a measurement of CORINE's
generalisation, not of the classifier.

The prediction map makes the same point visually — the model resolves the road
network and settlement fingers at 10 m, detail CORINE cannot represent at all,
and is scored down for it:

![Prediction maps](data/results/figures/fig5_prediction_maps.jpg)

### 3. Aggregate area agreement is not spatial agreement

From PostGIS, predicted vs CORINE area for artificial surfaces agrees to
**1.5%** (88.6 vs 87.3 km²) — while disagreeing spatially across the whole
scene. A report quoting only per-class areas would look far better than the
classification deserves.

---

## Per-class results, including the failures

![Confusion matrix](data/results/figures/fig1_confusion_matrix.png)

| Class | Precision | Recall | F1 | Share of AOI |
|---|---:|---:|---:|---:|
| Artificial surfaces | 0.801 | 0.818 | **0.809** | 19.8% |
| Broad-leaved forest | 0.662 | 0.906 | **0.761** | 29.3% |
| Pastures | 0.488 | 0.650 | 0.556 | 16.4% |
| Arable land | 0.555 | 0.557 | 0.553 | 13.8% |
| Heterogeneous agriculture | 0.286 | 0.116 | **0.165** | 14.1% |
| Mixed forest | 0.464 | 0.021 | **0.039** | 4.2% |
| Coniferous forest | 0.234 | 0.011 | **0.021** | 2.3% |

Three classes fail, each for a different and identifiable reason:

- **Heterogeneous agriculture** (CLC 242/243) is a *mosaic* class — "complex
  cultivation patterns", "agriculture with significant natural vegetation". It
  describes a mixture, not a surface. The confusion matrix shows it going 46% to
  pastures and 22% to arable, which is arguably the model being right about what
  each pixel physically is while being scored wrong.
- **Coniferous and mixed forest** are 2.3% and 4.2% of the area and sit next to a
  class covering 29%. In a single late-August scene, at peak leaf-on, they are
  swallowed by broad-leaved forest (79% and 91% respectively). Multi-date
  imagery — a winter scene, where deciduous canopy is bare and conifers are not
  — is the fix, not a better classifier.
- **Wetlands** were dropped before training: 0.10% of the AOI in a single
  polygon. A class confined to one polygon cannot be evaluated under a
  spatially-blocked split, because it lands entirely inside one fold.

`class_weight='balanced'` was measured rather than guessed: macro F1 rises
0.415 → 0.468 and coniferous recall 0.011 → 0.356, but overall accuracy falls
0.624 → 0.577 and predicted per-class *areas* degrade. The unbalanced model is
kept because area per class is what the PostGIS stage reports.

### What the forest actually uses

![Feature importance](data/results/figures/fig4_feature_importance.png)

SWIR (B11/B12) and the SWIR-based NBR2 dominate; **NDVI barely registers**. In a
late-August scene almost everything vegetated has saturated NDVI, so it cannot
separate forest from pasture from crops. That is an argument for carrying the
20 m SWIR bands rather than defaulting to the 10 m RGB+NIR that convenience
suggests.

---

## Pipeline

```bash
uv sync                                          # install
uv run python -m src.ingest.sentinel2 --list     # show candidate scenes
uv run python -m src.ingest.sentinel2            # fetch + preprocess (~40 s)
uv run python -m src.ingest.preview              # quicklook figure
uv run python -m src.labels.corine               # CORINE -> 10 m label raster
uv run python -m src.labels.preview              # label/imagery overlay check
uv run python -m src.models.random_forest        # train + evaluate (~5 min)
uv run python -m src.evaluation.figures          # result figures

docker compose -f docker/docker-compose.yml up -d
uv run python -m src.postprocessing.to_postgis   # load + spatial analysis
uv run python -m src.postprocessing.qgis_styles  # QGIS .qml styles
uv run pytest                                    # 68 tests
```

### 1. Imagery — `src/ingest/sentinel2.py`

Searches the [Element84 Earth Search](https://earth-search.aws.element84.com/v1)
STAC API and reads **only the AOI window** out of the public Cloud-Optimised
GeoTIFFs in `s3://sentinel-cogs` over HTTP range requests. No credentials, and
no 1 GB scene download for a 21 km subset — the whole ingest takes ~40 seconds.

Scene selection requires a single granule to **fully contain** the AOI rather
than mosaicking, so the stack never mixes acquisition dates (and therefore sun
angle and phenology) across the area. Selected: `S2A_31UGR_20240824_0_L2A`,
0.04% cloud, 99.998% valid pixels.

Output: 10 bands (B02–B12, the 20 m bands resampled to 10 m) plus NDVI, NDWI,
NDBI, NDRE and NBR2, as float32 with masked pixels as `NaN`.

### 2. Labels — `src/labels/corine.py`

The Copernicus Land Monitoring Service download portal requires an account, so
the same CLC2018 **vector** product is fetched anonymously from the EEA's public
ArcGIS REST service in its native EPSG:3035, then reprojected and rasterised
onto the imagery grid. CLC level 3 (44 classes) is aggregated to 12, of which 7
occur here.

![CORINE label overlay](data/results/figures/corine_labels_overlay.jpg)

### 3. Model — `src/models/random_forest.py`, `src/evaluation/`

300 trees, `min_samples_leaf=20`, `max_features='sqrt'`, 300 k training pixels,
no spatial context. Deliberately a pure per-pixel spectral baseline: it answers
"how much land cover is readable straight out of the spectrum?" and leaves a
clean, measurable gap for a convolutional model to close.

### 4. Database — `src/postprocessing/to_postgis.py`, `sql/`

The classified raster is polygonised (6,570 polygons ≥ 2000 m²) and loaded with
the communes, the CORINE reference and the AOI footprint, all in EPSG:32631 so
`ST_Area` returns square metres without a geography cast.

The AOI footprint is a table rather than a comment for a reason: without it,
per-commune areas are quietly wrong. A commune half outside the scene reports
half its true forest with nothing in the output to reveal it, so `commune_aoi`
carries `covered_fraction` alongside every area and the queries filter on it.

Selected output (`data/results/postgis_analysis.txt`):

| Commune | Forest | Built | Forest % |
|---|---:|---:|---:|
| Steinsel | 13.88 km² | 2.81 km² | 66.8% |
| Niederanven | 23.64 km² | 7.61 km² | 60.0% |
| Luxembourg | 14.95 km² | 26.32 km² | 30.4% |

---

## Design decisions worth defending

**The output CRS is read from the scene, never hardcoded.** Luxembourg straddles
the UTM 31N/32N boundary at 6°E. The obvious answer, EPSG:32632, is *wrong* for
this AOI — only MGRS tile 31UGR (EPSG:**32631**) fully contains it. That is the
kind of assumption that silently produces a shifted map.

**Labels are reprojected to the imagery, not the other way round.** Reprojecting
10 m imagery resamples every pixel and smears exactly the class boundaries the
model is trying to predict. Reprojecting vector polygons is exact up to the
datum transformation, and rasterisation then happens once, deliberately, with
the pixel-centre rule (`all_touched=False`) so adjacent polygons cannot
overwrite each other along shared edges.

**The target grid is snapped to the 10 m band grid with even offsets and
extents,** so the 20 m bands upsample with no sub-pixel shift. Verified by
cross-correlating native B08 against upsampled B8A: the peak sits at exactly
(0, 0) and falls off symmetrically (`tests/test_alignment.py`).

**The BOA reflectance offset is handled explicitly.** From processing baseline
04.00, ESA adds +1000 to L2A reflectance so slightly-negative retrievals survive
unsigned encoding. Earth Search flags whether it already removed it. Getting
this wrong shifts every band by 0.1 reflectance — enough to wreck NDVI and make
two scenes silently incomparable.

**Masked pixels are `NaN`, not 0.** Zero is a plausible reflectance and would be
learned as a feature value.

**Topographic shadow and "unclassified" pixels are kept.** Dropping SCL classes
2 and 7 would systematically remove north-facing slopes and genuinely ambiguous
ground, biasing the training set toward easy pixels and the score upward.

**NDMI was removed after being written.** It is identically `−NDBI`, which would
hand the forest two perfectly anti-correlated features and split its importance
scores across a duplicate. `tests/test_sentinel2.py` now asserts that no index
pair exceeds |r| = 0.99, so it cannot come back.

---

## Limitations

- **Single date.** One August 2024 scene. Coniferous/mixed forest and the
  arable/pasture distinction need multi-date imagery to work properly.
- **Labels are 2018, imagery is 2024.** Six years of real change is scored as
  model error. The "built-up where CORINE is not" query (2.3 km² in Luxembourg
  commune) partly measures that and partly measures CORINE's 25 ha MMU.
- **CORINE's 25 ha MMU.** The median polygon is 78 ha. Rivers, hedgerows and
  individual buildings do not exist in the labels. There is **no water class at
  all** in this AOI — the Alzette is narrower than the MMU.
- **One AOI, one scene.** Nothing here demonstrates generalisation to another
  region or season; the blocked split only demonstrates generalisation to unseen
  ground *within* this scene.
- **No spatial context in the model.** A per-pixel classifier produces
  salt-and-pepper noise, and no CRF or morphological filter has been applied.
- **Speckle filtering discards area.** Polygons below 2000 m² are dropped before
  loading to PostGIS; the loader reports how much area that removes rather than
  letting the totals silently not add up.

## Not done

- **CNN / U-Net + CRF post-processing.** The direct link to my thesis and the
  natural next step, since it would add the spatial context the Random Forest
  lacks. Not written. Given the boundary-distance analysis above, I would expect
  it to improve the map's appearance more than its CORINE-measured score,
  because CORINE cannot resolve the boundaries a CNN would sharpen.
- **QGIS print layout.** Styles are generated (`qgis/*.qml`) and the layout steps
  documented in `qgis/README.md`, but QGIS is not installed on the machine this
  was built on, so no exported layout image is included.

## Data sources and licensing

- **Sentinel-2 L2A** — Copernicus, via AWS Open Data / Element84 Earth Search.
  *Contains modified Copernicus Sentinel data 2024.*
- **CORINE Land Cover 2018** — © European Environment Agency. Copernicus data,
  free and open under Regulation (EU) No 1159/2013.
- **Commune boundaries** — Administration du cadastre et de la topographie, via
  [data.public.lu](https://data.public.lu).

Raw and processed rasters are not committed (see `.gitignore`); every artefact
is reproducible from the commands above.

## Repository layout

```
src/ingest/         STAC search, windowed COG reads, masking, indices
src/labels/         CORINE fetch, reprojection, rasterisation
src/models/         Random Forest baseline
src/evaluation/     spatial splitting, metrics, figures
src/postprocessing/ PostGIS loading, QGIS style generation
sql/                schema + spatial analysis queries
tests/              68 tests (pure logic + integration against a real stack)
docker/             PostGIS container
qgis/               generated .qml styles + layout instructions
```
