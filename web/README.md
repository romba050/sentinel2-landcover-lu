# Interactive web demo

A fully static, self-contained demo of the project: a swipe comparison between
the Sentinel-2 scene and the classification (with a toggle between the raw
per-pixel Random Forest and the mean-field-CRF-smoothed map — the thesis
method), plus a per-commune forest choropleth, with the honest headline
numbers alongside.

Everything in this folder is generated or vendored — there is **no server, no
database and no external request** at view time (even the map library is
vendored), so the page cannot break or stall during a demo.

## View it locally

Double-click `index.html`, or from the repo root:

```bash
open web/index.html            # macOS
# or, if a browser is picky about file:// URLs:
python3 -m http.server -d web 8000   # then http://localhost:8000
```

Deep links: `index.html#choro` opens directly in choropleth mode;
`index.html#raw` opens the swipe with the raw (un-smoothed) classification;
`index.html#corine` opens it with the CORINE 2018 reference layer.

## Put it on your website

All asset paths are relative, so the folder works from any subdirectory of any
static host. Two options:

**As its own page** — copy the folder into your site, e.g. as
`basile-rommes.com/landcover/`:

```bash
rsync -av web/ /path/to/your-site/landcover/
```

**Embedded in an existing page** — host the folder as above, then iframe it:

```html
<iframe src="/landcover/" style="width:100%;height:640px;border:1px solid #e4e3de;border-radius:8px"
        title="Sentinel-2 land cover classification, Luxembourg" loading="lazy"></iframe>
```

Total payload is ~2.3 MB (mostly the two rasters), well within normal
page-weight budgets.

## Regenerate after a pipeline change

```bash
uv run python -m src.postprocessing.web_export
```

This rebuilds `truecolor.webp`, `classification.png` and `data.js` from the
current pipeline outputs. `index.html` reads everything (legend, stats,
commune stats, bounds) from `data.js`, so the page can never drift from the
model results. `vendor/` is Leaflet 1.9.4, vendored deliberately.

## Files

| File | Origin |
|---|---|
| `index.html` | hand-written page (map logic, layout, swipe) |
| `data.js` | generated — bounds, legend, metrics, commune GeoJSON |
| `truecolor.webp` | generated — stretched scene, warped to EPSG:3857 |
| `classification_rf.png` | generated — out-of-fold RF map, class colours, warped |
| `classification_crf.png` | generated — the same probabilities after mean-field CRF |
| `classification_corine.png` | generated — the CORINE 2018 reference labels, same palette |
| `vendor/leaflet.{js,css}` | Leaflet 1.9.4, vendored |

The displayed pair is deliberately out-of-fold unary vs the CRF on top of it:
the identical model underneath, so every visible difference between the two
classification layers is the smoothing and nothing else. Commune statistics
are computed from the CRF map (the end product a per-commune report would use).

Note the rasters here are warped to Web-Mercator *for display only* — the
analysis rasters in `data/` remain in the scene's native UTM CRS. The warp is
what makes the overlays line up with the (correctly projected) commune
boundaries; see the docstring in `src/postprocessing/web_export.py`.
