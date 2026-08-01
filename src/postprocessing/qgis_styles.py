"""Generate QGIS layer styles (.qml) from the project's class definition.

The colours live in exactly one place: ``src.labels.corine.CLASSES``. The label
raster, the prediction raster, the PostGIS polygons and the matplotlib figures
all derive from it, so a class can never end up magenta in QGIS and green in the
report. Hand-maintaining a .qml alongside the Python is how that drift starts.

    uv run python -m src.postprocessing.qgis_styles

Load in QGIS:  Layer Properties -> Symbology -> Style -> Load Style -> the .qml
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from src.labels.corine import CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
QGIS_DIR = PROJECT_ROOT / "qgis"

QGIS_VERSION = "3.34"
DOCTYPE = "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>"


def _rgba(hex_colour: str, alpha: int = 255) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"{r},{g},{b},{alpha}"


def kept_classes(processed_dir: Path = PROCESSED):
    """The classes actually present, in label-scheme order.

    Falls back to the full scheme if the labels have not been built yet, so the
    styles can be generated on a fresh clone.
    """
    manifest = processed_dir / "latest_labels.json"
    if not manifest.exists():
        return [(c.code, c.name, c.colour) for c in CLASSES]
    meta = json.loads(manifest.read_text())
    return [
        (c["class_id"], c["name"], c["colour"])
        for c in meta["class_scheme"]["classes"]
        if c["kept"]
    ]


def raster_style(classes) -> str:
    """A paletted (discrete) raster renderer -- never a continuous ramp.

    Class ids are nominal: 6 is not "more" than 3. A continuous colour ramp on
    a class raster invents an ordering that does not exist and makes adjacent
    ids look related when they are not.
    """
    qgis = ET.Element("qgis", version=QGIS_VERSION, styleCategories="AllStyleCategories")
    pipe = ET.SubElement(qgis, "pipe")
    renderer = ET.SubElement(
        pipe, "rasterrenderer",
        type="paletted", band="1", opacity="1", alphaBand="-1", nodataColor="",
    )
    ET.SubElement(renderer, "rasterTransparency")
    palette = ET.SubElement(renderer, "colorPalette")
    for value, name, colour in classes:
        ET.SubElement(
            palette, "paletteEntry",
            value=str(value), color=colour, label=name, alpha="255",
        )
    ET.SubElement(qgis, "blendMode").text = "0"
    return DOCTYPE + "\n" + ET.tostring(qgis, encoding="unicode")


def vector_style(classes, attribute: str = "class_id") -> str:
    """A categorised fill renderer for the PostGIS polygon layer."""
    qgis = ET.Element("qgis", version=QGIS_VERSION, styleCategories="Symbology")
    renderer = ET.SubElement(
        qgis, "renderer-v2",
        type="categorizedSymbol", attr=attribute,
        forceraster="0", symbollevels="0", enableorderby="0", referencescale="-1",
    )
    categories = ET.SubElement(renderer, "categories")
    symbols = ET.SubElement(renderer, "symbols")
    for i, (value, name, colour) in enumerate(classes):
        ET.SubElement(
            categories, "category",
            render="true", value=str(value), symbol=str(i), label=name,
        )
        symbol = ET.SubElement(
            symbols, "symbol",
            type="fill", name=str(i), alpha="1", clip_to_extent="1", force_rhr="0",
        )
        layer = ET.SubElement(
            symbol, "layer", **{"class": "SimpleFill", "locked": "0",
                                "pass": "0", "enabled": "1"},
        )
        options = ET.SubElement(layer, "Option", type="Map")
        # No outline: at 10 m resolution every polygon edge is a pixel
        # staircase, and stroking them turns the map into a mesh of hairlines
        # that hides the fills the reader is actually meant to compare.
        for key, val in (
            ("color", _rgba(colour)),
            ("style", "solid"),
            ("outline_style", "no"),
            ("outline_color", _rgba(colour)),
        ):
            ET.SubElement(options, "Option", name=key, type="QString", value=val)
    return DOCTYPE + "\n" + ET.tostring(qgis, encoding="unicode")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=QGIS_DIR)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    classes = kept_classes(args.processed_dir)
    written = {
        "landcover_raster.qml": raster_style(classes),
        "landcover_polygons.qml": vector_style(classes),
    }
    for name, xml in written.items():
        (args.out_dir / name).write_text(xml)
        print(f"wrote {(args.out_dir / name).relative_to(PROJECT_ROOT)}")
    print(f"  {len(classes)} classes: " + ", ".join(n for _, n, _ in classes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
