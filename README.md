from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import folium
from folium.plugins import HeatMap
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber

COLUMN_BINS = [
    ("region", 0, 180),
    ("province", 180, 300),
    ("municipality", 300, 430),
    ("barangay", 430, 600),
    ("vhl", 600, 687),
    ("hl", 687, 758),
    ("ml", 758, 831),
    ("df", 831, 890),
    ("vhf", 890, 970),
    ("hf", 970, 1045),
    ("mf", 1045, 1195),
]

RISK_COLORS = {
    "Very High": "#d7301f",
    "High": "#fc8d59",
    "Moderate": "#fdbb2d",
}


def clean_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_name(value) -> str:
    value = clean_text(value).lower()
    value = value.replace("(capital)", "")
    value = value.replace("city of ", "")
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def municipality_key(region: str, province: str, municipality: str) -> str:
    return "|".join(
        [normalize_name(region), normalize_name(province), normalize_name(municipality)]
    )


def extract_report_meta(pdf_bytes: bytes) -> Dict[str, str]:
    meta = {"report_date": "", "report_time": ""}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = pdf.pages[0].extract_text() or ""
    date_match = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
    time_match = re.search(r"\b(\d{4})\b", text)
    if date_match:
        meta["report_date"] = date_match.group(1)
    if time_match:
        raw = time_match.group(1)
        if len(raw) == 4:
            meta["report_time"] = f"{raw[:2]}:{raw[2:]}"
    return meta


def _group_page_words(page) -> list[list[dict]]:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    groups, current = [], []
    last_top = None
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if last_top is None or abs(word["top"] - last_top) <= 1.5:
            current.append(word)
            last_top = word["top"] if last_top is None else (last_top + word["top"]) / 2
        else:
            groups.append(current)
            current = [word]
            last_top = word["top"]
    if current:
        groups.append(current)
    return groups


def extract_pdf_rows(pdf_bytes: bytes) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for group in _group_page_words(page):
                top = min(w["top"] for w in group)
                text_row = " ".join(w["text"] for w in group)
                if top < 66:
                    continue
                if text_row.startswith("Barangay Count:") or text_row.startswith("Note:"):
                    continue
                if text_row.strip().isdigit() and len(group) == 1:
                    continue

                rec = {"page": page_no, "top": top}
                for name, xmin, xmax in COLUMN_BINS:
                    ws = [
                        w
                        for w in group
                        if xmin <= ((w["x0"] + w["x1"]) / 2) < xmax
                    ]
                    rec[name] = " ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"]))
                if any(rec[c] for c in ["region", "province", "municipality", "barangay"]):
                    rows.append(rec)
    return pd.DataFrame(rows)


def prepare_landslide_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for c in ["region", "province", "municipality", "barangay", "vhl", "hl", "ml"]:
        df[c] = df[c].map(clean_text)

    def classify(row) -> str | None:
        if row["vhl"] == "VHL":
            return "Very High"
        if row["hl"] == "HL":
            return "High"
        if row["ml"] == "ML":
            return "Moderate"
        return None

    df["landslide_risk"] = df.apply(classify, axis=1)
    df = df[df["landslide_risk"].notna()].copy()
    df["weight"] = df["landslide_risk"].map({"Very High": 3, "High": 2, "Moderate": 1})
    return df.reset_index(drop=True)


def summarize_municipalities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["region", "province", "municipality"], as_index=False)
        .agg(
            affected_barangays=("barangay", "count"),
            very_high=("landslide_risk", lambda s: (s == "Very High").sum()),
            high=("landslide_risk", lambda s: (s == "High").sum()),
            moderate=("landslide_risk", lambda s: (s == "Moderate").sum()),
            score=("weight", "sum"),
        )
    )

    def dominant(row) -> str:
        if row["very_high"] > 0:
            return "Very High"
        if row["high"] > 0:
            return "High"
        return "Moderate"

    summary["dominant_risk"] = summary.apply(dominant, axis=1)
    summary["key"] = summary.apply(
        lambda r: municipality_key(r["region"], r["province"], r["municipality"]), axis=1
    )
    return summary.sort_values(["affected_barangays", "score"], ascending=[False, False])


def _empty_cache_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["key", "region", "province", "municipality", "query", "lat", "lon", "source"]
    )


def load_cache(cache_path: str | Path) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        return _empty_cache_df()
    df = pd.read_csv(path)
    expected = ["key", "region", "province", "municipality", "query", "lat", "lon", "source"]
    for col in expected:
        if col not in df.columns:
            df[col] = "" if col not in ["lat", "lon"] else pd.NA
    return df[expected]


def save_cache(cache_df: pd.DataFrame, cache_path: str | Path) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(path, index=False)


def ensure_municipality_coordinates(
    summary_df: pd.DataFrame,
    cache_path: str | Path = "cache/municipality_cache.csv",
    user_agent: str = "mgb-osm-visualizer",
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    cache_df = load_cache(cache_path)
    cache_df = cache_df.drop_duplicates(subset=["key"], keep="last")
    merged = summary_df.merge(cache_df[["key", "lat", "lon", "source"]], on="key", how="left")

    missing = merged[merged["lat"].isna() | merged["lon"].isna()].copy()
    if not missing.empty:
        geolocator = Nominatim(user_agent=user_agent)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

        new_rows = []
        for _, row in missing.iterrows():
            query = f"{row['municipality']}, {row['province']}, Philippines"
            loc = geocode(query)
            if loc is not None:
                lat, lon = loc.latitude, loc.longitude
                source = "nominatim"
            else:
                lat, lon, source = pd.NA, pd.NA, "unmatched"
            new_rows.append(
                {
                    "key": row["key"],
                    "region": row["region"],
                    "province": row["province"],
                    "municipality": row["municipality"],
                    "query": query,
                    "lat": lat,
                    "lon": lon,
                    "source": source,
                }
            )

        if new_rows:
            cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
            cache_df = cache_df.drop_duplicates(subset=["key"], keep="last")
            save_cache(cache_df, cache_path)
            merged = summary_df.merge(cache_df[["key", "lat", "lon", "source"]], on="key", how="left")

    return merged


def build_folium_map(geo_df: pd.DataFrame, title: str = "Municipality landslide hotspots") -> folium.Map:
    map_df = geo_df.dropna(subset=["lat", "lon"]).copy()
    if map_df.empty:
        return folium.Map(location=[12.8797, 121.7740], zoom_start=5, tiles="OpenStreetMap")

    center = [map_df["lat"].mean(), map_df["lon"].mean()]
    m = folium.Map(location=center, zoom_start=8, tiles="OpenStreetMap", control_scale=True)

    max_score = max(map_df["score"].max(), 1)
    heat_data = [
        [row.lat, row.lon, float(row.score) / max_score]
        for _, row in map_df.iterrows()
    ]
    HeatMap(heat_data, radius=30, blur=22, min_opacity=0.3).add_to(m)

    for _, row in map_df.iterrows():
        color = RISK_COLORS.get(row["dominant_risk"], "#fdbb2d")
        radius = 6 + (18 * float(row["score"]) / max_score)
        popup = folium.Popup(
            (
                f"<b>{row['municipality']}</b><br>"
                f"{row['province']}<br>"
                f"Affected barangays: {int(row['affected_barangays'])}<br>"
                f"VHL: {int(row['very_high'])} | HL: {int(row['high'])} | ML: {int(row['moderate'])}<br>"
                f"Score: {int(row['score'])}"
            ),
            max_width=320,
        )
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=radius,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            popup=popup,
            tooltip=f"{row['municipality']} ({int(row['affected_barangays'])})",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def municipality_summary_geojson(geo_df: pd.DataFrame) -> str:
    features = []
    for _, row in geo_df.dropna(subset=["lat", "lon"]).iterrows():
        props = row.drop(labels=["lat", "lon"]).to_dict()
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
                "properties": props,
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2)


def make_facebook_png(geo_df: pd.DataFrame, scope_label: str, meta: Dict[str, str]) -> bytes:
    buf = io.BytesIO()
    fig = plt.figure(figsize=(7.2, 9.0), dpi=150, facecolor="white")
    ax_map = fig.add_axes([0.07, 0.18, 0.60, 0.70])
    ax_side = fig.add_axes([0.71, 0.20, 0.24, 0.66])
    ax_footer = fig.add_axes([0.07, 0.06, 0.88, 0.07])
    ax_side.axis("off")
    ax_footer.axis("off")

    fig.text(0.07, 0.95, scope_label, fontsize=20, weight="bold", color="#0b2e59")
    subtitle = "Rain-induced landslide municipality hotspot map"
    if meta.get("report_date") or meta.get("report_time"):
        subtitle += f" • {meta.get('report_date', '')} {meta.get('report_time', '')}".rstrip()
    fig.text(0.07, 0.92, subtitle, fontsize=9, color="#40566b")

    plot_df = geo_df.dropna(subset=["lat", "lon"]).copy()
    ax_map.set_facecolor("#ecf5fb")
    if not plot_df.empty:
        xmin, xmax = plot_df["lon"].min(), plot_df["lon"].max()
        ymin, ymax = plot_df["lat"].min(), plot_df["lat"].max()
        xpad = max((xmax - xmin) * 0.20, 0.12)
        ypad = max((ymax - ymin) * 0.20, 0.12)
        ax_map.set_xlim(xmin - xpad, xmax + xpad)
        ax_map.set_ylim(ymin - ypad, ymax + ypad)
        max_score = max(plot_df["score"].max(), 1)
        cmap = plt.cm.get_cmap("YlOrRd")
        norm = plot_df["score"] / max_score
        colors = [cmap(max(0.20, float(v))) for v in norm]

        for size, alpha in [(2300, 0.07), (1200, 0.12), (500, 0.22)]:
            ax_map.scatter(
                plot_df["lon"],
                plot_df["lat"],
                s=(plot_df["score"] / max_score) * size + 50,
                c=colors,
                alpha=alpha,
                linewidths=0,
            )
        ax_map.scatter(
            plot_df["lon"],
            plot_df["lat"],
            s=(plot_df["score"] / max_score) * 65 + 18,
            c=colors,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

        top_labels = plot_df.sort_values(["affected_barangays", "score"], ascending=False).head(10)
        for _, row in top_labels.iterrows():
            ax_map.text(
                row["lon"] + 0.02,
                row["lat"] + 0.02,
                row["municipality"],
                fontsize=7,
                color="#16263a",
                zorder=4,
            )
    else:
        ax_map.text(0.5, 0.5, "No matched coordinates available", ha="center", va="center")

    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#b8d1e5")
        spine.set_linewidth(1.0)

    total_brgys = int(geo_df["affected_barangays"].sum()) if not geo_df.empty else 0
    total_munis = int(len(geo_df))
    ax_side.text(0, 1.00, "Summary", fontsize=14, weight="bold", color="#0b2e59")
    ax_side.text(0, 0.93, f"Affected barangays\n{total_brgys}", fontsize=12, weight="bold", color="#0b2e59")
    ax_side.text(0, 0.84, f"Affected municipalities\n{total_munis}", fontsize=12, weight="bold", color="#0b2e59")

    vhl = int(geo_df["very_high"].sum()) if "very_high" in geo_df else 0
    hl = int(geo_df["high"].sum()) if "high" in geo_df else 0
    ml = int(geo_df["moderate"].sum()) if "moderate" in geo_df else 0
    ax_side.text(0, 0.73, "Risk mix", fontsize=12, weight="bold", color="#0b2e59")
    ax_side.text(0.02, 0.68, f"Very High: {vhl}", color=RISK_COLORS["Very High"], fontsize=10)
    ax_side.text(0.02, 0.64, f"High: {hl}", color=RISK_COLORS["High"], fontsize=10)
    ax_side.text(0.02, 0.60, f"Moderate: {ml}", color=RISK_COLORS["Moderate"], fontsize=10)

    ax_side.text(0, 0.52, "Top municipalities", fontsize=12, weight="bold", color="#0b2e59")
    top10 = geo_df.sort_values(["affected_barangays", "score"], ascending=False).head(10)
    y = 0.48
    for i, (_, row) in enumerate(top10.iterrows(), start=1):
        ax_side.text(0.00, y, f"{i}. {row['municipality']}", fontsize=9, color="#16263a")
        ax_side.text(0.98, y, str(int(row['affected_barangays'])), fontsize=9, color="#16263a", ha="right")
        y -= 0.04

    ax_footer.text(
        0,
        0.72,
        "Municipality-level hotspot visualization based on MGB barangay list. Colors show relative concentration of listed landslide-prone barangays.",
        fontsize=7.5,
        color="#41576d",
    )
    ax_footer.text(
        0,
        0.30,
        "Approximate visualization only; this is not an official MGB hazard polygon map.",
        fontsize=7.5,
        color="#41576d",
    )

    plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
