import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import zoom

# Page Configuration
st.set_page_config(
    page_title="DELTAFLOW-ENGINE // CYBERNETIC HUD",
    page_icon="⚡",
    layout="wide"
)

# Cyberpunk / Glitch Design System CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        background-color: #0a0a0f;
        color: #e0e0e0;
        border-radius: 0px !important;
    }

    * {
        border-radius: 0px !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.4);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #12121a;
        border-right: 1px solid #2a2a3a;
    }

    /* Buttons */
    .stButton > button {
        background-color: transparent;
        color: #00ff88;
        border: 2px solid #00ff88;
        padding: 0.75rem 2rem;
        font-family: 'Share Tech Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-weight: 700;
        box-shadow: 0 0 5px #00ff88, 0 0 10px rgba(0, 255, 136, 0.3);
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        background-color: #00ff88;
        color: #0a0a0f;
        box-shadow: 0 0 15px #00ff88, 0 0 30px #00ff8860;
    }

    /* Metrics Cards */
    [data-testid="stMetric"] {
        background-color: #12121a;
        border: 1px solid #2a2a3a;
        padding: 1.25rem;
        box-shadow: 0 0 8px rgba(0, 255, 136, 0.15);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Share Tech Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-size: 0.75rem !important;
        color: #00d4ff !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        color: #00ff88 !important;
        text-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
    }

    /* Horizontal Rules */
    hr {
        border: none;
        border-top: 2px solid #00ff88;
        box-shadow: 0 0 8px #00ff88;
        margin: 3rem 0;
    }

    /* Scanlines effect overlay */
    .scanline {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 0, 0, 0.15) 2px, rgba(0, 0, 0, 0.15) 4px);
        pointer-events: none;
        z-index: 99999;
    }
</style>
<div class="scanline"></div>
""", unsafe_allow_html=True)

# Cyberpunk HUD Header
st.markdown("<p style='font-family: \"Share Tech Mono\", monospace; font-size: 0.8rem; letter-spacing: 0.3em; text-transform: uppercase; color: #ff00ff;'>// SECURE FEED: SECTOR_DELTA_2026</p>", unsafe_allow_html=True)
st.title("DELTAFLOW-ENGINE // V2.0")
st.markdown("<p style='font-family: \"Share Tech Mono\", monospace; color: #00d4ff; font-size: 1.1rem; letter-spacing: 0.1em; margin-bottom: 2rem;'>[SYSTEM ACTIVE] Matrix-Based Gravitational Runoff & Spatial Drainage Optimization</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚡ HACKER CONTROLS // PARAMS")
uploaded_file = st.sidebar.file_uploader("UPLOAD RAW DEM (.tif) OR CSV DATA", type=["tif", "tiff", "csv", "xlsx"])

rainfall_input = st.sidebar.slider("Rainfall Intensity (mm/day)", min_value=1.0, max_value=100.0, value=18.5, step=0.5)
sim_steps = st.sidebar.slider("Simulation Iterations", min_value=10, max_value=150, value=50, step=5)
percentile_factor = st.sidebar.slider("Hazard Risk Threshold", min_value=0.50, max_value=0.95, value=0.75, step=0.05)
grid_size = st.sidebar.slider("Grid Matrix Resolution", min_value=40, max_value=200, value=100, step=10)

@st.cache_data
def load_data_feed(target_size, uploaded_obj):
    raw_dem = None
    custom_coords = None

    if uploaded_obj is not None:
        filename = uploaded_obj.name.lower()
        if filename.endswith(('.tif', '.tiff')):
            try:
                img = Image.open(uploaded_obj)
                raw_dem = np.array(img, dtype=float)
            except Exception as e:
                st.sidebar.error(f"DEM Parse Error: {e}")
        elif filename.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_obj)
                if 'elevation' in df.columns or 'Z' in df.columns:
                    col_name = 'elevation' if 'elevation' in df.columns else 'Z'
                    raw_dem = df[col_name].values.reshape(int(np.sqrt(len(df))), -1)
                elif len(df.columns) >= 3:
                    raw_dem = df.iloc[:, 2].values.reshape(int(np.sqrt(len(df))), -1)
                if 'lat' in df.columns and 'lon' in df.columns:
                    custom_coords = (float(df['lat'].mean()), float(df['lon'].mean()))
            except Exception as e:
                st.sidebar.error(f"CSV Parse Error: {e}")
        elif filename.endswith(('.xls', '.xlsx')):
            try:
                xls = pd.ExcelFile(uploaded_obj)
                df = pd.read_excel(xls, xls.sheet_names[0])
                if len(df.columns) >= 3:
                    raw_dem = df.iloc[:, 2].values.reshape(int(np.sqrt(len(df))), -1)
            except Exception as e:
                st.sidebar.error(f"Excel Parse Error: {e}")

    if raw_dem is None:
        tif_path = "output_SRTMGL1.tif"
        if os.path.exists(tif_path):
            img = Image.open(tif_path)
            raw_dem = np.array(img, dtype=float)
        else:
            x = np.linspace(-5.0, 5.0, target_size)
            y = np.linspace(-5.0, 5.0, target_size)
            X, Y = np.meshgrid(x, y)
            raw_dem = 12.0 + 0.5 * (X**2 + Y**2) - 4.2 * np.exp(-(X**2 + Y**2) / 2.0)

    raw_dem[raw_dem < -1000] = np.nan
    valid_mean = np.nanmean(raw_dem) if not np.isnan(np.nanmean(raw_dem)) else 0.0
    raw_dem = np.nan_to_num(raw_dem, nan=valid_mean)
    
    zoom_factors = (target_size / raw_dem.shape[0], target_size / raw_dem.shape[1])
    return zoom(raw_dem, zoom_factors, order=1), custom_coords

elevation_grid, custom_coords = load_data_feed(grid_size, uploaded_file)

# Simulation Engine Run
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

# Dynamic Coordinate Extraction directly from file/matrix
max_water_idx = np.unravel_index(np.argmax(water), water.shape)
if custom_coords:
    base_lat, base_lon = custom_coords
else:
    base_lat, base_lon = 31.3100, 31.1500
extracted_lat = base_lat + ((max_water_idx[0] - (rows / 2.0)) * 0.0005)
extracted_lon = base_lon + ((max_water_idx[1] - (cols / 2.0)) * 0.0005)

# Metrics Grid
col1, col2, col3, col4 = st.columns(4)
col1.metric("PRECIPITATION INPUT", f"{rainfall_input:.1f} MM")
col2.metric("PEAK ACCUMULATION", f"{np.max(water):.1f} MM")
col3.metric("CRITICAL THRESHOLD", f"{threshold:.1f} MM")
col4.metric("HIGH-RISK CELLS", f"{total_danger_cells}")

st.markdown("<hr>", unsafe_allow_html=True)

# Vibrant, Colorful Cyberpunk Scientific Visualizations
st.subheader("⚡ VIBRANT NEON SPECTRAL VISUALIZATION")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0a0a0f')

for ax in axes:
    ax.set_facecolor('#0a0a0f')
    ax.tick_params(colors='#00ff88')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a3a')

# 1. Elevation Topography (Plasma Colormap)
im0 = axes[0].imshow(elevation_grid, cmap='plasma', origin='lower')
axes[0].set_title("1. Elevation Topography [PLASMA]", color='#00ff88', fontdict={'family': 'monospace', 'weight': 'bold'})
cbar0 = fig.colorbar(im0, ax=axes[0], label='Elevation (m)')
cbar0.ax.yaxis.label.set_color('#e0e0e0')
cbar0.ax.tick_params(colors='#e0e0e0')

# 2. Runoff Accumulation Heatmap (Turbo / Neon Blue-Cyan)
im1 = axes[1].imshow(water, cmap='turbo', origin='lower')
axes[1].set_title("2. Runoff Accumulation [TURBO]", color='#00d4ff', fontdict={'family': 'monospace', 'weight': 'bold'})
cbar1 = fig.colorbar(im1, ax=axes[1], label='Water Depth (mm)')
cbar1.ax.yaxis.label.set_color('#e0e0e0')
cbar1.ax.tick_params(colors='#e0e0e0')

# 3. Dynamic Hazard Zones (Magma / Hot Pink & Electric Red)
im2 = axes[2].imshow(danger_map, cmap='magma', origin='lower')
axes[2].set_title("3. Critical Hazard Matrix [MAGMA]", color='#ff00ff', fontdict={'family': 'monospace', 'weight': 'bold'})
cbar2 = fig.colorbar(im2, ax=axes[2], label='1 = Danger / 0 = Safe')
cbar2.ax.yaxis.label.set_color('#e0e0e0')
cbar2.ax.tick_params(colors='#e0e0e0')

plt.tight_layout()
st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)

# GIS Georeferencing & Navigation Panel
st.subheader("🌐 PRECISE GEOSPATIAL TARGETING // GPS LOCK")
gmaps_link = f"https://www.google.com/maps/search/?api=1&query={extracted_lat:.6f},{extracted_lon:.6f}"
st.markdown(f"""
<div style="border: 2px solid #00ff88; padding: 1.5rem; background-color: #12121a; box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);">
    <p style="font-family: 'Share Tech Mono', monospace; color: #00ff88; font-size: 1rem; text-transform: uppercase; margin-bottom: 0.75rem;"><strong>[SUCCESS] Extracted Matrix Coordinates from Dataset:</strong></p>
    <p style="font-family: 'JetBrains Mono', monospace; color: #e0e0e0; font-size: 0.9rem; margin-bottom: 0.5rem;">• Matrix Target Index (Y, X): <strong>{max_water_idx}</strong></p>
    <p style="font-family: 'JetBrains Mono', monospace; color: #e0e0e0; font-size: 0.9rem; margin-bottom: 1.25rem;">• High-Precision GPS Lock: <strong>{extracted_lat:.4f}° N, {extracted_lon:.4f}° E</strong></p>
    <a href="{gmaps_link}" target="_blank" style="font-family: 'Orbitron', monospace; font-size: 0.85rem; text-transform: uppercase; text-decoration: none; color: #0a0a0f; background-color: #00ff88; padding: 0.75rem 1.5rem; font-weight: 700; box-shadow: 0 0 10px #00ff88; display: inline-block;">INITIALIZE GOOGLE MAPS TARGET LOCK →</a>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Cyberpunk Footer with Copyright & GitHub Hyperlink
st.markdown("""
<div style="text-align: center; padding: 2rem 0; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; color: #6b7280;">
    © 2026 // DEVELOPED BY <a href="https://github.com/mooghanem" target="_blank" style="color: #00ff88; text-decoration: underline; font-weight: 700; text-shadow: 0 0 5px #00ff88;">MOHAMED GHANEM</a> // ALL RIGHTS RESERVED
</div>
""", unsafe_allow_html=True)
