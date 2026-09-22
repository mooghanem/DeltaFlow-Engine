import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import zoom

# Page Configuration
st.set_page_config(
    page_title="DeltaFlow-Engine | Global DEM Simulator",
    page_icon="▪️",
    layout="wide"
)

# Minimalist Monochrome Custom CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Source+Serif+4:ital,opsz,wght@0,8..60,200..900;1,8..60,200..900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Serif 4', Georgia, serif;
        background-color: #FFFFFF;
        color: #000000;
        border-radius: 0px !important;
    }

    /* Force zero border radius everywhere */
    * {
        border-radius: 0px !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #000000;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5;
        border-right: 1px solid #000000;
    }

    /* Buttons */
    .stButton > button {
        background-color: #000000;
        color: #FFFFFF;
        border: none;
        padding: 0.75rem 2rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
        transition: background-color 100ms ease, color 100ms ease;
    }
    .stButton > button:hover {
        background-color: #FFFFFF;
        color: #000000;
        border: 2px solid #000000;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        border: 1px solid #000000;
        padding: 1.25rem;
        background-color: #FFFFFF;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.75rem !important;
        color: #525252 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 700;
        color: #000000 !important;
    }

    /* Horizontal Rules */
    hr {
        border: none;
        border-top: 4px solid #000000;
        margin: 3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<p style='font-family: \"JetBrains Mono\", monospace; font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; color: #525252;'>System Monograph // 2026</p>", unsafe_allow_html=True)
st.title("DeltaFlow-Engine")
st.markdown("<p style='font-size: 1.25rem; font-style: italic; margin-bottom: 2rem;'>A Matrix-Based Universal Elevation & Urban Flood Simulation Engine</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar Parameters & DEM Input
st.sidebar.header("Simulation Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Global DEM (.tif, .tiff)", type=["tif", "tiff"])

rainfall_input = st.sidebar.slider("Rainfall Intensity (mm/day)", min_value=1.0, max_value=100.0, value=15.0, step=0.5)
sim_steps = st.sidebar.slider("Simulation Steps", min_value=10, max_value=120, value=40, step=5)
percentile_factor = st.sidebar.slider("Hazard Threshold Percentile", min_value=0.50, max_value=0.95, value=0.75, step=0.05)
grid_size = st.sidebar.slider("Grid Resolution (pixels)", min_value=40, max_value=200, value=100, step=10)

@st.cache_data
def load_global_dem(target_size, uploaded_obj):
    if uploaded_obj is not None:
        try:
            img = Image.open(uploaded_obj)
            raw_dem = np.array(img, dtype=float)
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {e}")
            raw_dem = None
    else:
        tif_path = "output_SRTMGL1.tif"
        if os.path.exists(tif_path):
            img = Image.open(tif_path)
            raw_dem = np.array(img, dtype=float)
        else:
            raw_dem = None

    if raw_dem is None:
        x = np.linspace(-5.0, 5.0, target_size)
        y = np.linspace(-5.0, 5.0, target_size)
        X, Y = np.meshgrid(x, y)
        return 15.0 + 0.4 * (X**2 + Y**2) - 4.0 * np.exp(-(X**2 + Y**2) / 2.0)

    raw_dem[raw_dem < -1000] = np.nan
    valid_mean = np.nanmean(raw_dem) if not np.isnan(np.nanmean(raw_dem)) else 0.0
    raw_dem = np.nan_to_num(raw_dem, nan=valid_mean)
    
    zoom_factors = (target_size / raw_dem.shape[0], target_size / raw_dem.shape[1])
    return zoom(raw_dem, zoom_factors, order=1)

elevation_grid = load_global_dem(grid_size, uploaded_file)

# Run surface runoff simulation
rows, cols = elevation_grid.shape
water = np.ones((rows, cols)) * (rainfall_input / 10.0)
for _ in range(sim_steps):
    dy, dx = np.gradient(elevation_grid + water)
    flow_x = -dx * 0.18
    flow_y = -dy * 0.18
    divergence = np.gradient(flow_x, axis=1) + np.gradient(flow_y, axis=0)
    water -= divergence * 0.5
    water = np.clip(water, 0.0, None)
    water += (rainfall_input / 100.0)

min_w, max_w = np.min(water), np.max(water)
threshold = min_w + percentile_factor * (max_w - min_w)
danger_map = (water >= threshold).astype(int)
total_danger_cells = int(np.sum(danger_map))

# Dynamic coordinate extraction
max_water_idx = np.unravel_index(np.argmax(water), water.shape)
base_lat, base_lon = 31.3100, 31.1500
extracted_lat = base_lat + ((max_water_idx[0] - (rows / 2.0)) * 0.0005)
extracted_lon = base_lon + ((max_water_idx[1] - (cols / 2.0)) * 0.0005)

# Metrics display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rainfall Input", f"{rainfall_input:.1f} mm/day")
col2.metric("Max Water Accum.", f"{np.max(water):.1f} mm")
col3.metric("Critical Threshold", f"{threshold:.1f} mm")
col4.metric("High-Risk Cells", f"{total_danger_cells}")

st.markdown("<hr>", unsafe_allow_html=True)

# Visualizations
st.subheader("Simulation Visualizations")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#FFFFFF')

for ax in axes:
    ax.set_facecolor('#FFFFFF')

im0 = axes[0].imshow(elevation_grid, cmap='gray', origin='lower')
axes[0].set_title("1. Elevation Topography", fontdict={'family': 'serif', 'weight': 'bold'})
fig.colorbar(im0, ax=axes[0], label='Elevation (m)')

im1 = axes[1].imshow(water, cmap='gray', origin='lower')
axes[1].set_title("2. Computed Runoff Heatmap", fontdict={'family': 'serif', 'weight': 'bold'})
fig.colorbar(im1, ax=axes[1], label='Water Depth (mm)')

im2 = axes[2].imshow(danger_map, cmap='gray', origin='lower')
axes[2].set_title("3. Dynamic Hazard Zones", fontdict={'family': 'serif', 'weight': 'bold'})
fig.colorbar(im2, ax=axes[2], label='1 = Danger / 0 = Safe')

plt.tight_layout()
st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)

# GIS & Google Maps integration
st.subheader("Spatial Georeferencing & Navigation")
gmaps_link = f"https://www.google.com/maps/search/?api=1&query={extracted_lat:.6f},{extracted_lon:.6f}"
st.markdown(f"""
<div style="border: 2px solid #000000; padding: 1.5rem; background-color: #FFFFFF;">
    <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; text-transform: uppercase; margin-bottom: 0.5rem;"><strong>Active Grid Index (Y, X):</strong> {max_water_idx}</p>
    <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; text-transform: uppercase; margin-bottom: 1rem;"><strong>Extracted GPS Coordinates:</strong> {extracted_lat:.4f}° N, {extracted_lon:.4f}° E</p>
    <a href="{gmaps_link}" target="_blank" style="font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; text-transform: uppercase; text-decoration: underline; color: #000000; font-weight: 700;">Open Target Zone in Google Maps →</a>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Footer conforming strictly to Minimalist Monochrome design & user request
st.markdown("""
<div style="text-align: center; padding: 2rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: #525252;">
    © 2026 Made by <a href="https://github.com/mooghanem" target="_blank" style="color: #000000; text-decoration: underline; font-weight: 700;">Mohamed Ghanem</a>. All rights reserved.
</div>
""", unsafe_allow_html=True)
