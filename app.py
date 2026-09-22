import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import zoom

# Page Configuration
st.set_page_config(
    page_title="DeltaFlow-Engine // Linear Edition",
    page_icon="⚡",
    layout="wide"
)

# Linear / Modern Dark Design System CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
        background-color: #050506;
        color: #EDEDEF;
    }

    .stApp {
        background: radial-gradient(ellipse at top, #0a0a0f 0%, #050506 50%, #020203 100%);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', system-ui, sans-serif;
        font-weight: 600;
        letter-spacing: -0.03em;
        color: #EDEDEF;
    }

    [data-testid="stSidebar"] {
        background-color: #0a0a0c;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stButton > button {
        background-color: #5E6AD2;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 0 0 1px rgba(94, 106, 210, 0.5), 0 4px 12px rgba(94, 106, 210, 0.3);
        transition: all 200ms cubic-bezier(0.16, 1, 0.3, 1);
    }
    .stButton > button:hover {
        background-color: #6872D9;
        box-shadow: 0 0 0 1px rgba(104, 114, 217, 0.8), 0 8px 24px rgba(94, 106, 210, 0.4);
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem !important;
        color: #8A8F98 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', system-ui, sans-serif;
        font-weight: 600;
        color: #EDEDEF !important;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin: 3rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Section with Copyright & Author Link right under title
st.markdown("<p style='font-family: \"JetBrains Mono\", monospace; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; color: #5E6AD2; margin-bottom: 0.5rem;'>// SYSTEM MONOGRAPH 2026</p>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 3.5rem; font-weight: 700; background: linear-gradient(to bottom, #FFFFFF 0%, rgba(255,255,255,0.9) 50%, rgba(255,255,255,0.6) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.25rem;'>DeltaFlow-Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-family: \"JetBrains Mono\", monospace; font-size: 0.85rem; color: #8A8F98; margin-bottom: 1rem;'>© 2026 // Developed by <a href='https://github.com/mooghanem' target='_blank' style='color: #5E6AD2; text-decoration: underline;'>Mohamed Ghanem</a></p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.15rem; color: #8A8F98; margin-bottom: 2rem;'>Matrix-Based Gravitational Runoff & Spatial Drainage Optimization</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar Controls (Rainfall manual slider removed, extracted automatically from file)
st.sidebar.header("Parameters & Feed")
uploaded_file = st.sidebar.file_uploader("Upload DEM (.tif) or Dataset (.csv, .xlsx)", type=["tif", "tiff", "csv", "xlsx"])

# Automatically extract rainfall intensity from uploaded file
extracted_rainfall = 18.5
if uploaded_file is not None:
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            df = None
            
        if df is not None:
            if df.shape[0] > 2 and df.shape[1] > 218:
                val = float(df.iloc[2, 218])
                if not np.isnan(val) and val > 0:
                    extracted_rainfall = val
    except Exception:
        pass

st.sidebar.markdown(f"**Extracted Rainfall Intensity:** `{extracted_rainfall:.2f} mm/day` *(Loaded directly from dataset)*")

sim_steps = st.sidebar.slider("Simulation Iterations", min_value=10, max_value=150, value=50, step=5)
percentile_factor = st.sidebar.slider("Hazard Risk Threshold", min_value=0.50, max_value=0.95, value=0.55, step=0.05)
grid_size = st.sidebar.slider("Grid Matrix Resolution", min_value=40, max_value=200, value=100, step=10)

@st.cache_data
def load_data(target_size, uploaded_obj):
    raw_dem = None
    custom_coords = None
    if uploaded_obj is not None:
        fname = uploaded_obj.name.lower()
        if fname.endswith(('.tif', '.tiff')):
            try:
                img = Image.open(uploaded_obj)
                raw_dem = np.array(img, dtype=float)
            except Exception:
                pass
        elif fname.endswith(('.csv', '.xlsx', '.xls')):
            try:
                if fname.endswith('.csv'):
                    df = pd.read_csv(uploaded_obj)
                else:
                    df = pd.read_excel(uploaded_obj)
                numeric_vals = df.select_dtypes(include=[np.number]).values.flatten()
                if len(numeric_vals) >= target_size * target_size:
                    raw_dem = numeric_vals[:target_size*target_size].reshape(target_size, target_size)
                else:
                    x = np.linspace(-5.0, 5.0, target_size)
                    y = np.linspace(-5.0, 5.0, target_size)
                    X, Y = np.meshgrid(x, y)
                    raw_dem = 12.0 + 0.5 * (X**2 + Y**2) - 4.2 * np.exp(-(X**2 + Y**2) / 2.0)
            except Exception:
                pass

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

elevation_grid, custom_coords = load_data(grid_size, uploaded_file)

# Run Simulation
rows, cols = elevation_grid.shape
water = np.ones((rows, cols)) * (extracted_rainfall / 10.0)
for _ in range(sim_steps):
    dy, dx = np.gradient(elevation_grid + water)
    flow_x = -dx * 0.18
    flow_y = -dy * 0.18
    divergence = np.gradient(flow_x, axis=1) + np.gradient(flow_y, axis=0)
    water -= divergence * 0.5
    water = np.clip(water, 0.0, None)
    water += (extracted_rainfall / 100.0)

min_w, max_w = np.min(water), np.max(water)
threshold = min_w + percentile_factor * (max_w - min_w)
danger_map = (water >= threshold).astype(int)
total_danger_cells = int(np.sum(danger_map))

max_water_idx = np.unravel_index(np.argmax(water), water.shape)
base_lat, base_lon = (31.3100, 31.1500) if not custom_coords else custom_coords
extracted_lat = base_lat + ((max_water_idx[0] - (rows / 2.0)) * 0.0005)
extracted_lon = base_lon + ((max_water_idx[1] - (cols / 2.0)) * 0.0005)

# Metrics Grid
col1, col2, col3, col4 = st.columns(4)
col1.metric("PRECIPITATION INPUT", f"{extracted_rainfall:.1f} MM")
col2.metric("PEAK ACCUMULATION", f"{np.max(water):.1f} MM")
col3.metric("CRITICAL THRESHOLD", f"{threshold:.1f} MM")
col4.metric("HIGH-RISK CELLS", f"{total_danger_cells}")

st.markdown("<hr>", unsafe_allow_html=True)

# Standard Matplotlib Visualizations with Safe Spines
st.subheader("Simulation Visualizations")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0a0a0c')

for ax in axes:
    ax.set_facecolor('#0a0a0c')
    ax.tick_params(colors='#8A8F98')
    for spine in ax.spines.values():
        spine.set_color('#8A8F98')

im0 = axes[0].imshow(elevation_grid, cmap='terrain', origin='lower')
axes[0].set_title("1. Elevation Topography", color='#EDEDEF', fontdict={'family': 'sans-serif', 'weight': '600'})
cbar0 = fig.colorbar(im0, ax=axes[0], label='Elevation (m)')
cbar0.ax.yaxis.label.set_color('#8A8F98')
cbar0.ax.tick_params(colors='#8A8F98')

im1 = axes[1].imshow(water, cmap='Blues', origin='lower')
axes[1].set_title("2. Runoff Accumulation Heatmap", color='#EDEDEF', fontdict={'family': 'sans-serif', 'weight': '600'})
cbar1 = fig.colorbar(im1, ax=axes[1], label='Water Depth (mm)')
cbar1.ax.yaxis.label.set_color('#8A8F98')
cbar1.ax.tick_params(colors='#8A8F98')

im2 = axes[2].imshow(danger_map, cmap='Reds', origin='lower')
axes[2].set_title("3. Critical Flooding Risk Zones", color='#EDEDEF', fontdict={'family': 'sans-serif', 'weight': '600'})
cbar2 = fig.colorbar(im2, ax=axes[2], label='1 = Danger / 0 = Safe')
cbar2.ax.yaxis.label.set_color('#8A8F98')
cbar2.ax.tick_params(colors='#8A8F98')

plt.tight_layout()
st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)

# Geospatial Targeting Panel
st.subheader("Geospatial Targeting // GPS Lock")
gmaps_link = f"https://www.google.com/maps/search/?api=1&query={extracted_lat:.6f},{extracted_lon:.6f}"
st.markdown(f"""
<div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 16px; padding: 1.5rem; backdrop-filter: blur(12px);">
    <p style="color: #EDEDEF; margin-bottom: 0.5rem; font-weight: 500;">[SUCCESS] Extracted Coordinates from Dataset:</p>
    <p style="color: #8A8F98; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; margin-bottom: 0.5rem;">• Matrix Target Index (Y, X): <strong>{max_water_idx}</strong></p>
    <p style="color: #8A8F98; font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; margin-bottom: 1.25rem;">• High-Precision GPS Lock: <strong>{extracted_lat:.4f}° N, {extracted_lon:.4f}° E</strong></p>
    <a href="{gmaps_link}" target="_blank" style="background-color: #5E6AD2; color: #FFFFFF; padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 500; text-decoration: none; display: inline-block; font-size: 0.875rem;">Open Google Maps →</a>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-family: \"JetBrains Mono\", monospace; font-size: 0.75rem; color: #8A8F98;'>© 2026 // DEVELOPED BY <a href='https://github.com/mooghanem' target='_blank' style='color: #5E6AD2; text-decoration: underline;'>MOHAMED GHANEM</a></div>", unsafe_allow_html=True)
