from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from mgb_core import (
    build_folium_map,
    ensure_municipality_coordinates,
    extract_pdf_rows,
    extract_report_meta,
    make_facebook_png,
    municipality_summary_geojson,
    prepare_landslide_dataframe,
    summarize_municipalities,
)

st.set_page_config(page_title="MGB Landslide Municipality Visualizer", page_icon="⛰️", layout="wide")

st.title("MGB Landslide Municipality Visualizer")
st.caption(
    "Upload an MGB barangay-list PDF. The app extracts landslide entries, groups them by municipality, "
    "places municipality hotspots on OpenStreetMap, and generates a Facebook-friendly PNG automatically."
)

CACHE_PATH = "cache/municipality_cache.csv"


@st.cache_data(show_spinner=False)
def parse_pdf(pdf_bytes: bytes):
    raw = extract_pdf_rows(pdf_bytes)
    landslide = prepare_landslide_dataframe(raw)
    meta = extract_report_meta(pdf_bytes)
    return landslide, meta


uploaded = st.file_uploader("Upload MGB PDF", type=["pdf"])

if not uploaded:
    st.info("Upload an MGB PDF to start.")
    st.stop()

pdf_bytes = uploaded.getvalue()

with st.spinner("Reading PDF and extracting landslide entries..."):
    landslide_df, report_meta = parse_pdf(pdf_bytes)

if landslide_df.empty:
    st.error("No landslide entries (VHL, HL, ML) were found in the uploaded PDF.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    regions = ["All"] + sorted(landslide_df["region"].dropna().unique().tolist())
    region = st.selectbox("Region", regions)

    df1 = landslide_df.copy()
    if region != "All":
        df1 = df1[df1["region"] == region]

    provinces = ["All"] + sorted(df1["province"].dropna().unique().tolist())
    province = st.selectbox("Province", provinces)

    df2 = df1.copy()
    if province != "All":
        df2 = df2[df2["province"] == province]

    risk_choices = ["Very High", "High", "Moderate"]
    risks = st.multiselect("Include susceptibility", risk_choices, default=risk_choices)

filtered = df2[df2["landslide_risk"].isin(risks)].copy()
if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

summary = summarize_municipalities(filtered)
scope_parts = []
if region != "All":
    scope_parts.append(region)
if province != "All":
    scope_parts.append(province)
scope_label = " • ".join(scope_parts) if scope_parts else "Philippines"

with st.spinner("Preparing municipality centroids and map..."):
    geo_summary = ensure_municipality_coordinates(summary, cache_path=CACHE_PATH)

matched = geo_summary.dropna(subset=["lat", "lon"])
unmatched = geo_summary[geo_summary["lat"].isna() | geo_summary["lon"].isna()]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Affected barangays", int(filtered.shape[0]))
m2.metric("Affected municipalities", int(summary.shape[0]))
m3.metric("With coordinates", int(matched.shape[0]))
m4.metric("Unmatched municipalities", int(unmatched.shape[0]))

col_map, col_side = st.columns([1.9, 1.1])

with col_map:
    st.subheader(f"OpenStreetMap hotspot view — {scope_label}")
    fmap = build_folium_map(geo_summary, title=scope_label)
    st_folium(fmap, width=None, height=680)
    if not unmatched.empty:
        with st.expander("Show unmatched municipalities"):
            st.dataframe(unmatched[["region", "province", "municipality", "affected_barangays"]], use_container_width=True)

with col_side:
    st.subheader("Municipality summary")
    st.dataframe(
        geo_summary[
            [
                "region",
                "province",
                "municipality",
                "affected_barangays",
                "very_high",
                "high",
                "moderate",
                "score",
                "dominant_risk",
            ]
        ],
        use_container_width=True,
        height=680,
    )

st.subheader("Downloads")
png_bytes = make_facebook_png(geo_summary, scope_label=scope_label, meta=report_meta)
geojson_text = municipality_summary_geojson(geo_summary)
csv_bytes = geo_summary.to_csv(index=False).encode("utf-8")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Download municipality summary CSV",
        data=csv_bytes,
        file_name="mgb_municipality_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c2:
    st.download_button(
        "Download municipality GeoJSON",
        data=geojson_text,
        file_name="mgb_municipality_summary.geojson",
        mime="application/geo+json",
        use_container_width=True,
    )
with c3:
    st.download_button(
        "Download Facebook PNG (1080x1350)",
        data=png_bytes,
        file_name="mgb_facebook_hotspot_map.png",
        mime="image/png",
        use_container_width=True,
    )

st.info(
    "This app is municipality-focused. The hotspot glow represents the relative concentration of listed landslide-prone barangays within each municipality. "
    "It is an approximate visualization, not an official MGB hazard polygon map."
)
