"""
DeltaFlow-Engine
A Matrix-Based Digital Elevation and Gravitational Runoff Simulation Engine
for Urban Flood Mitigation in Deltaic Regions.

Case study site: El-Hamoul, Kafr El-Sheikh, Egypt (31.3100 N, 31.1500 E)

WHAT THIS FILE FIXES vs. the version in the repo right now
------------------------------------------------------------
1. "Zero Desert Error" was only a slogan before: the old code actually
   invented a fake `file_seed` from the uploaded filename and used it to
   randomly shift base_lat/base_lon and to fabricate a synthetic sine-wave
   terrain whenever anything other than a .tif was uploaded. That is the
   opposite of locking coordinates -- it's what caused pins to drift into
   the desert. This version always simulates on the real SRTM DEM and only
   ever changes the elevation surface if you explicitly upload a new
   GeoTIFF, in which case it checks the file's own coordinates against a
   Nile-Delta bounding box and warns you loudly instead of silently
   accepting it.
2. GPS coordinates used to be `base_lat + index * 0.0005` -- an arbitrary
   made-up constant with no relationship to the DEM's real pixel size.
   This version reads the GeoTIFF's real affine transform with rasterio
   and converts matrix indices to true latitude/longitude.
3. Rainfall extraction used to be `df.iloc[2, 218]` -- a magic cell that
   only works for one specific NASA POWER export layout and silently
   returns nonsense (or crashes) for any other file. This version searches
   the uploaded file for likely precipitation columns, shows you a preview,
   and lets you confirm/pick the right cell instead of guessing blindly.

WHERE TO GET REAL INPUT DATA
------------------------------------------------------------
- SRTM elevation (GeoTIFF): https://portal.opentopography.org/raster?opentopoID=OTSRTM.042013.4326.1
  (search "SRTM GL1 30m", draw a box around 31.20-31.42 N / 31.05-31.25 E,
  export as GeoTIFF). The repo already ships `output_SRTMGL1.tif` for this.
- NASA POWER rainfall: https://power.larc.nasa.gov/data-access-viewer/
  Pick "Agroclimatology" or "Climatology" community, parameter
  PRECTOTCORR (bias-corrected precipitation), point = 31.31, 31.15,
  and export CSV or XLSX. The "Climatic Design Conditions" report format
  (percentile table) and the plain daily time-series format are both
  handled below.
"""

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom

try:
    import rasterio
    from rasterio.io import MemoryFile
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# Fixed site constants -- the real, permanent anchor for this case study.
# These are NEVER derived from an uploaded filename or a random seed.
# ---------------------------------------------------------------------------
SITE_NAME = "El-Hamoul, Kafr El-Sheikh, Egypt"
SITE_LAT = 31.3100
SITE_LON = 31.1500
DEFAULT_DEM_PATH = "output_SRTMGL1.tif"
DEFAULT_RAINFALL_MM_DAY = 13.20  # fallback design-storm value if nothing can be parsed
FRICTION_COEFF_DEFAULT = 0.18
SIM_GRID_RES_DEFAULT = 100

# Loose bounding box around the wider Nile Delta, used only as a sanity
# check on uploaded DEMs -- not to force-fit coordinates.
DELTA_BBOX = dict(lat_min=28.5, lat_max=32.5, lon_min=29.0, lon_max=33.0)

st.set_page_config(page_title="DeltaFlow-Engine // Linear Edition", page_icon="\u26a1", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; background-color: #050506; color: #EDEDEF; }
    .stApp { background: radial-gradient(ellipse at top, #0a0a0f 0%, #050506 50%, #020203 100%); }
    h1, h2, h3, h4, h5, h6 { font-family: 'Inter', system-ui, sans-serif; font-weight: 600; letter-spacing: -0.03em; color: #EDEDEF; }
    [data-testid="stSidebar"] { background-color: #0a0a0c; border-right: 1px solid rgba(255,255,255,0.06); }
    .stButton > button { background-color: #5E6AD2; color: #FFFFFF; border: none; border-radius: 8px; padding: 0.6rem 1.5rem; font-weight: 500; box-shadow: 0 0 0 1px rgba(94,106,210,0.5), 0 4px 12px rgba(94,106,210,0.3); }
    [data-testid="stMetric"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 1.25rem; backdrop-filter: blur(12px); }
    [data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem !important; color: #8A8F98 !important; }
    [data-testid="stMetricValue"] { font-weight: 600; color: #EDEDEF !important; }
    hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<p style='font-family:\"JetBrains Mono\",monospace;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:#5E6AD2;'>// SYSTEM MONOGRAPH 2026</p>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size:3.2rem;font-weight:700;margin-bottom:0.25rem;'>DeltaFlow-Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-family:\"JetBrains Mono\",monospace;font-size:0.85rem;color:#8A8F98;'>\u00a9 2026 // Developed by <a href='https://github.com/mooghanem' target='_blank' style='color:#5E6AD2;text-decoration:underline;'>Mohamed Ghanem</a></p>", unsafe_allow_html=True)
st.markdown(f"<p style='font-size:1.05rem;color:#8A8F98;'>Matrix-Based Gravitational Runoff & Spatial Drainage Optimization &mdash; case study: {SITE_NAME}</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if not HAS_RASTERIO:
    st.warning(
        "`rasterio` is not installed, so real GeoTIFF georeferencing is unavailable and this run "
        "falls back to an approximate linear coordinate estimate anchored at the fixed site "
        "coordinates. Install it with `pip install rasterio` for accurate GPS output."
    )

# ---------------------------------------------------------------------------
# DEM loading -- always real data, never fabricated terrain.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dem(file_bytes, is_default):
    """Load a GeoTIFF and return (elevation_array, transform, crs, bounds).

    transform/crs/bounds are None only when rasterio isn't installed; the
    elevation array itself always comes from the real file, never from a
    generated function.
    """
    if HAS_RASTERIO:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                arr = src.read(1).astype(float)
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
        return arr, transform, crs, bounds
    else:
        img = Image.open(io_bytes(file_bytes))
        arr = np.array(img, dtype=float)
        return arr, None, None, None


def io_bytes(b):
    import io
    return io.BytesIO(b)


def dem_center_lonlat(transform, arr_shape):
    """Real center coordinate of a georeferenced DEM, from its own transform."""
    if transform is None:
        return None, None
    rows, cols = arr_shape
    lon, lat = rasterio.transform.xy(transform, rows // 2, cols // 2)
    return lat, lon


def pixel_to_lonlat(row, col, transform):
    if transform is None:
        return None, None
    lon, lat = rasterio.transform.xy(transform, row, col)
    return lat, lon


st.sidebar.header("Elevation Source")
dem_upload = st.sidebar.file_uploader(
    "Optional: override the default DEM with your own GeoTIFF (.tif)",
    type=["tif", "tiff"],
    help="Leave empty to simulate on the case-study SRTM tile shipped with the repo.",
)

if dem_upload is not None:
    raw_bytes = dem_upload.getvalue()
    is_default = False
else:
    if os.path.exists(DEFAULT_DEM_PATH):
        with open(DEFAULT_DEM_PATH, "rb") as f:
            raw_bytes = f.read()
        is_default = True
    else:
        raw_bytes = None
        is_default = True

if raw_bytes is not None:
    raw_dem, dem_transform, dem_crs, dem_bounds = load_dem(raw_bytes, is_default)
else:
    st.error(f"Default DEM `{DEFAULT_DEM_PATH}` not found and no file was uploaded. Cannot run a real simulation.")
    st.stop()

raw_dem[raw_dem < -1000] = np.nan
valid_mean = np.nanmean(raw_dem) if not np.isnan(np.nanmean(raw_dem)) else 0.0
raw_dem = np.nan_to_num(raw_dem, nan=valid_mean)

# Sanity-check any *uploaded* DEM against the Nile Delta bounding box instead
# of silently trusting or silently randomizing it.
center_lat, center_lon = dem_center_lonlat(dem_transform, raw_dem.shape)
if not is_default and center_lat is not None:
    if not (DELTA_BBOX["lat_min"] <= center_lat <= DELTA_BBOX["lat_max"] and
            DELTA_BBOX["lon_min"] <= center_lon <= DELTA_BBOX["lon_max"]):
        st.error(
            f"[ZERO DESERT ERROR CHECK] This GeoTIFF's own coordinates center on "
            f"{center_lat:.4f} N, {center_lon:.4f} E, which falls **outside** the expected "
            f"Nile Delta region. Proceeding would risk plotting results in the wrong place "
            f"(e.g. the Western Desert). Double-check the file before trusting the output below."
        )

base_lat = center_lat if center_lat is not None else SITE_LAT
base_lon = center_lon if center_lon is not None else SITE_LON

sim_grid_res = st.sidebar.slider("Grid Matrix Resolution", min_value=40, max_value=200, value=SIM_GRID_RES_DEFAULT, step=10)
zoom_factors = (sim_grid_res / raw_dem.shape[0], sim_grid_res / raw_dem.shape[1])
elevation_grid = zoom(raw_dem, zoom_factors, order=1)

# ---------------------------------------------------------------------------
# Rainfall extraction -- transparent, user-confirmed, no magic cell index.
# ---------------------------------------------------------------------------

st.sidebar.header("Rainfall Dataset")
rain_upload = st.sidebar.file_uploader(
    "Upload a NASA POWER export (.csv or .xlsx)",
    type=["csv", "xlsx", "xls"],
)

extracted_rainfall = DEFAULT_RAINFALL_MM_DAY
rainfall_source_note = f"design-storm fallback ({DEFAULT_RAINFALL_MM_DAY:.2f} mm/day) -- no dataset parsed"

if rain_upload is not None:
    fname = rain_upload.name.lower()
    try:
        if fname.endswith(".csv"):
            # NASA POWER point-CSV exports start with metadata lines ending in
            # "-END HEADER-"; find the real header row instead of assuming row 0.
            text = rain_upload.getvalue().decode("utf-8", errors="ignore")
            lines = text.splitlines()
            header_idx = 0
            for i, line in enumerate(lines):
                if "-END HEADER-" in line:
                    header_idx = i + 1
                    break
            from io import StringIO
            df = pd.read_csv(StringIO(text), skiprows=header_idx)
        else:
            df = pd.read_excel(rain_upload, sheet_name=0)

        # Find columns that look like precipitation.
        candidate_cols = [c for c in df.columns if re.search(r"PRECTOT|PRECIP|RAIN", str(c), re.IGNORECASE)]

        if not candidate_cols:
            st.sidebar.warning(
                "Couldn't auto-detect a rainfall column by name. Pick the right column and row manually below."
            )
            candidate_cols = list(df.columns)

        chosen_col = st.sidebar.selectbox("Rainfall column", candidate_cols, index=0)
        numeric_series = pd.to_numeric(df[chosen_col], errors="coerce").dropna()

        if len(numeric_series) == 0:
            raise ValueError("Selected column has no numeric values.")
        elif len(numeric_series) > 20:
            # Looks like a daily/hourly time series -- pick a design percentile.
            pctl = st.sidebar.slider("Design percentile", 50.0, 99.9, 99.6, 0.1)
            extracted_rainfall = float(np.percentile(numeric_series, pctl))
            rainfall_source_note = f"{pctl:.1f}th percentile of `{chosen_col}` ({len(numeric_series)} rows)"
        else:
            # Looks like a small percentile-indexed design table (e.g. NASA POWER
            # "Climatic Design Conditions" report) -- show it and let the user pick a row.
            label_col = df.columns[0]
            preview = df[[label_col, chosen_col]].copy()
            preview[chosen_col] = pd.to_numeric(preview[chosen_col], errors="coerce")
            st.sidebar.dataframe(preview, hide_index=True, height=180)
            row_choice = st.sidebar.selectbox(
                "Design row to use",
                preview.index,
                index=len(preview) - 1,
                format_func=lambda i: f"{preview.loc[i, label_col]} -> {preview.loc[i, chosen_col]}",
            )
            extracted_rainfall = float(preview.loc[row_choice, chosen_col])
            rainfall_source_note = f"row `{preview.loc[row_choice, label_col]}` of `{chosen_col}`"

    except Exception as e:
        st.sidebar.error(f"Could not parse `{rain_upload.name}` ({e}); using fallback value.")

st.sidebar.markdown(f"**Rainfall used:** `{extracted_rainfall:.2f} mm/day`")
st.sidebar.caption(rainfall_source_note)
st.sidebar.markdown(f"**Site coordinates:** `{base_lat:.4f}\u00b0 N, {base_lon:.4f}\u00b0 E`")

sim_steps = st.sidebar.slider("Simulation Iterations", min_value=10, max_value=150, value=35, step=5)
percentile_factor = st.sidebar.slider("Hazard Risk Threshold (percentile)", min_value=0.50, max_value=0.95, value=0.75, step=0.01)
friction_k = st.sidebar.slider("Friction Coefficient (k)", min_value=0.05, max_value=0.40, value=FRICTION_COEFF_DEFAULT, step=0.01)

# ---------------------------------------------------------------------------
# Numerical runoff engine -- simplified diffusive routing (not a full
# shallow-water solver; adequate for a screening-level ISEF demonstration).
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def process_simulation(elev, rain, steps, perc, k, dt=0.5):
    rows, cols = elev.shape
    water = np.ones((rows, cols)) * (rain / 10.0)
    for _ in range(steps):
        dy, dx = np.gradient(elev + water)
        flow_x = -dx * k
        flow_y = -dy * k
        divergence = np.gradient(flow_x, axis=1) + np.gradient(flow_y, axis=0)
        water -= divergence * dt
        water = np.clip(water, 0.0, None)
        water += (rain / 100.0)

    min_w, max_w = np.min(water), np.max(water)
    threshold = min_w + perc * (max_w - min_w)
    danger_map = (water >= threshold).astype(int)
    total_danger_cells = int(np.sum(danger_map))
    max_water_idx = np.unravel_index(np.argmax(water), water.shape)
    return water, danger_map, threshold, total_danger_cells, max_water_idx


water, danger_map, threshold, total_danger_cells, max_water_idx = process_simulation(
    elevation_grid, extracted_rainfall, sim_steps, percentile_factor, friction_k
)

# Map the peak-accumulation cell back to a REAL lat/lon using the DEM's own
# affine transform (scaled from the resized simulation grid to the original
# raster's index space), instead of an arbitrary `index * constant` formula.
orig_rows, orig_cols = raw_dem.shape
resized_rows, resized_cols = elevation_grid.shape
orig_row = max_water_idx[0] * (orig_rows / resized_rows)
orig_col = max_water_idx[1] * (orig_cols / resized_cols)

if dem_transform is not None:
    extracted_lat, extracted_lon = pixel_to_lonlat(orig_row, orig_col, dem_transform)
    gps_method = "real GeoTIFF affine transform"
else:
    # Fallback only: linear estimate anchored at the fixed site coordinates.
    extracted_lat = base_lat + ((max_water_idx[0] - (elevation_grid.shape[0] / 2.0)) * 0.0003)
    extracted_lon = base_lon + ((max_water_idx[1] - (elevation_grid.shape[1] / 2.0)) * 0.0003)
    gps_method = "approximate linear estimate (rasterio unavailable)"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("PRECIPITATION INPUT", f"{extracted_rainfall:.1f} MM")
col2.metric("PEAK ACCUMULATION", f"{np.max(water):.1f} MM")
col3.metric("CRITICAL THRESHOLD", f"{threshold:.1f} MM")
col4.metric("HIGH-RISK CELLS", f"{total_danger_cells}")

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

st.subheader("Simulation Visualizations")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0a0a0c')
for ax in axes:
    ax.set_facecolor('#0a0a0c')
    ax.tick_params(colors='#8A8F98')
    for spine in ax.spines.values():
        spine.set_color('#8A8F98')

im0 = axes[0].imshow(elevation_grid, cmap='terrain', origin='lower')
axes[0].set_title("1. Elevation Topography (real SRTM)", color='#EDEDEF')
fig.colorbar(im0, ax=axes[0], label='Elevation (m)')

im1 = axes[1].imshow(water, cmap='Blues', origin='lower')
axes[1].set_title("2. Runoff Accumulation Heatmap", color='#EDEDEF')
fig.colorbar(im1, ax=axes[1], label='Water Depth (mm)')

im2 = axes[2].imshow(danger_map, cmap='Reds', origin='lower')
axes[2].set_title("3. Critical Flooding Risk Zones", color='#EDEDEF')
fig.colorbar(im2, ax=axes[2], label='1 = Danger / 0 = Safe')

plt.tight_layout()
st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Geospatial targeting
# ---------------------------------------------------------------------------

st.subheader("Geospatial Targeting // GPS Lock")
gmaps_link = f"https://www.google.com/maps/search/?api=1&query={extracted_lat:.6f},{extracted_lon:.6f}"
st.markdown(f"""
<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 1.5rem; backdrop-filter: blur(12px);">
    <p style="color:#EDEDEF;margin-bottom:0.5rem;font-weight:500;">[SUCCESS] Extracted Coordinates:</p>
    <p style="color:#8A8F98;font-family:'JetBrains Mono',monospace;font-size:0.875rem;margin-bottom:0.25rem;">&bull; Matrix Target Index (Y, X): <strong>{max_water_idx}</strong></p>
    <p style="color:#8A8F98;font-family:'JetBrains Mono',monospace;font-size:0.875rem;margin-bottom:0.25rem;">&bull; High-Precision GPS Lock: <strong>{extracted_lat:.4f}\u00b0 N, {extracted_lon:.4f}\u00b0 E</strong></p>
    <p style="color:#5A5F68;font-size:0.75rem;margin-bottom:1rem;">Method: {gps_method}</p>
    <a href="{gmaps_link}" target="_blank" style="background-color:#5E6AD2;color:#FFFFFF;padding:0.6rem 1.2rem;border-radius:8px;font-weight:500;text-decoration:none;display:inline-block;font-size:0.875rem;">Open Google Maps &rarr;</a>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;font-family:\"JetBrains Mono\",monospace;font-size:0.75rem;color:#8A8F98;'>\u00a9 2026 DeltaFlow-Engine DEVELOPED BY <a href='https://github.com/mooghanem' target='_blank' style='color:#5E6AD2;text-decoration:underline;'>Mohamed Ghanem</a></div>", unsafe_allow_html=True)
