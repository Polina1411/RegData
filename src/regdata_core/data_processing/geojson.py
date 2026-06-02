from __future__ import annotations

import json
from pathlib import Path


def _simplify_ring(ring: list, decimals: int) -> list:
    simplified = []
    prev_point = None

    for point in ring:
        current_point = [round(point[0], decimals), round(point[1], decimals)]
        if current_point != prev_point:
            simplified.append(current_point)
            prev_point = current_point

    if len(simplified) >= 2 and simplified[0] != simplified[-1]:
        simplified.append(simplified[0])

    return simplified if len(simplified) >= 4 else ring


def _simplify_coordinates(coords: list, decimals: int) -> list:
    if not coords:
        return coords

    first = coords[0]
    if first and isinstance(first[0], (int, float)):
        return _simplify_ring(coords, decimals)

    return [_simplify_coordinates(item, decimals) for item in coords]


def build_lightweight_geojson(
    source_path: Path,
    output_path: Path,
    iso_key: str = "ISO3166-1-Alpha-3",
    decimals: int = 1,
) -> Path:
    # Для веб-карты храним облегчённую геометрию, чтобы интерфейс не тормозил на тяжёлом geojson.
    with source_path.open("r", encoding="utf-8") as f:
        raw_geojson = json.load(f)

    lightweight_features = []
    for feature in raw_geojson.get("features", []):
        properties = feature.get("properties", {})
        iso3 = properties.get(iso_key)
        geometry = feature.get("geometry")
        if not iso3 or not geometry:
            continue

        lightweight_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": geometry.get("type"),
                    "coordinates": _simplify_coordinates(
                        geometry.get("coordinates", []),
                        decimals=decimals,
                    ),
                },
                "properties": {
                    "iso3": iso3,
                    "name": properties.get("name", iso3),
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"type": "FeatureCollection", "features": lightweight_features},
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return output_path
