"""
preprocess.py
=============
Geospatial preprocessing pipeline for the Bajil Smart Village project.

Pipeline:
    1.  Load contour shapefile with geopandas
    2.  Auto-detect the elevation attribute field
    3.  Sample (x, y, z) points from contour geometries
    4.  Interpolate a regular grid using scipy’s griddata (linear method)
    5.  Write the result as a GeoTIFF (data/processed/dem.tif)
    6.  Print DEM metadata (CRS, shape, min/max elevation)
    7.  Visualise the DEM with Plotly (interactive)
    8.  Inspect an existing DEM (inspect_dem)
    9.  Clip DEM to boundary shapefile (clip_dem)
    10. Validate CRS for all layers (validate_crs)
    11. Check DEM quality — stats, spikes, nodata (check_dem_quality)
    12. Plot DEM for validation with matplotlib (plot_dem)
    13. Check boundary–DEM alignment (check_alignment)
"""

import sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from scipy.interpolate import griddata
from shapely.geometry import box  # ← استيراد أداة إنشاء الصناديق
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # …/project/
RAW_DIR           = PROJECT_ROOT / "data" / "raw"
PROC_DIR          = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR       = PROJECT_ROOT / "outputs" / "figures"   # ← مجلد الصور
CONTOUR_PATH      = RAW_DIR / "contor.shp"
DEM_OUT_PATH           = PROC_DIR / "dem.tif"
BOUNDARY_PATH          = RAW_DIR / "باب_الناقة.shp"
DEM_CLIPPED_PATH       = PROC_DIR / "dem_clipped.tif"
BOUNDARY_CLIPPED_PATH  = PROC_DIR / "boundary_clipped.shp"   # ← الملف الجديد المطلوب

# Output pixel resolution in map units (metres if CRS is projected)
TARGET_RESOLUTION = 10   # metres


# ---------------------------------------------------------------------------
# Helper: detect elevation field
# ---------------------------------------------------------------------------
def detect_elevation_field(gdf: gpd.GeoDataFrame) -> str:
    """
    Try to find the numeric field that represents elevation.

    Checks (case-insensitive) for common names first, then falls back to
    the first numeric column found.

    Returns
    -------
    str
        Name of the detected elevation column.

    Raises
    ------
    ValueError
        If no suitable field is found.
    """
    candidates = ["elev", "elevation", "height", "z", "alt",
                  "contour", "level", "hgt", "dem", "irt_cntr_v"]

    # Case-insensitive exact / partial match
    lower_cols = {c.lower(): c for c in gdf.columns}
    for candidate in candidates:
        if candidate in lower_cols:
            print(f"  [INFO] Elevation field detected: '{lower_cols[candidate]}'")
            return lower_cols[candidate]

    # Fallback: first numeric column that is not geometry
    for col in gdf.columns:
        if col == "geometry":
            continue
        if np.issubdtype(gdf[col].dtype, np.number):
            print(f"  [WARN] No standard elevation field found. "
                  f"Using first numeric column: '{col}'")
            return col

    raise ValueError(
        "Could not detect an elevation field in the shapefile. "
        f"Available columns: {list(gdf.columns)}"
    )


# ---------------------------------------------------------------------------
# Helper: sample (x, y, z) points from MultiLineString / LineString geometries
# ---------------------------------------------------------------------------
def sample_xyz_from_contours(gdf: gpd.GeoDataFrame,
                              elev_field: str) -> tuple:
    """
    Extract coordinate triples (x, y, z) from all vertices of the contour lines.

    Parameters
    ----------
    gdf : GeoDataFrame
        Contour GeoDataFrame.
    elev_field : str
        Name of the elevation attribute column.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        Arrays of x, y, z values sampled from every vertex.
    """
    xs, ys, zs = [], [], []

    for _, row in gdf.iterrows():
        z_val = float(row[elev_field])
        geom  = row.geometry

        if geom is None or geom.is_empty:
            continue

        # Support both LineString and MultiLineString
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            coords = np.array(part.coords)
            xs.append(coords[:, 0])
            ys.append(coords[:, 1])
            zs.append(np.full(len(coords), z_val))

    return (np.concatenate(xs),
            np.concatenate(ys),
            np.concatenate(zs))


# ---------------------------------------------------------------------------
# Helper: interpolate scattered points onto a regular grid
# ---------------------------------------------------------------------------
def interpolate_dem(x: np.ndarray,
                    y: np.ndarray,
                    z: np.ndarray,
                    resolution: float,
                    bounds: tuple) -> tuple:
    """
    Interpolate scattered (x, y, z) points to a regular grid.

    Parameters
    ----------
    x, y, z : np.ndarray
        Scattered sample coordinates and values.
    resolution : float
        Pixel size in map units.
    bounds : tuple
        (x_min, y_min, x_max, y_max) of the study area.

    Returns
    -------
    (grid_z, grid_x_edges, grid_y_edges)
        Interpolated elevation array and grid edges used for the transform.
    """
    x_min, y_min, x_max, y_max = bounds

    # Build grid cell centres
    grid_x = np.arange(x_min, x_max, resolution)
    grid_y = np.arange(y_max, y_min, -resolution)   # top-down (north first)

    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    print(f"  [INFO] Grid size: {grid_xx.shape[0]} rows × {grid_xx.shape[1]} cols")
    print(f"  [INFO] Interpolating {len(x):,} sample points …")

    grid_z = griddata(
        points=(x, y),
        values=z,
        xi=(grid_xx, grid_yy),
        method="linear"
    )

    # Fill any remaining NaN cells with nearest-neighbour
    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z_nn = griddata(
            points=(x, y),
            values=z,
            xi=(grid_xx, grid_yy),
            method="nearest"
        )
        grid_z[nan_mask] = grid_z_nn[nan_mask]
        print(f"  [INFO] Filled {nan_mask.sum():,} NaN cells using nearest-neighbour.")

    return grid_z, x_min, y_max   # top-left corner for rasterio transform


# ---------------------------------------------------------------------------
# Helper: write DEM GeoTIFF
# ---------------------------------------------------------------------------
def write_dem_tif(grid_z: np.ndarray,
                  x_min: float,
                  y_max: float,
                  resolution: float,
                  crs,
                  out_path: Path) -> None:
    """
    Write a 2-D elevation array to a single-band GeoTIFF.

    Parameters
    ----------
    grid_z : np.ndarray  shape (rows, cols)
    x_min, y_max : float
        Top-left corner coordinates.
    resolution : float
        Pixel size in map units.
    crs : rasterio.crs.CRS or str
        Coordinate reference system.
    out_path : Path
        Destination file path.
    """
    rows, cols = grid_z.shape
    transform = rasterio.transform.from_origin(x_min, y_max, resolution, resolution)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=grid_z.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999,
        compress="lzw",
    ) as dst:
        dst.write(grid_z.astype(grid_z.dtype), 1)

    print(f"  [INFO] DEM saved → {out_path}")


# ---------------------------------------------------------------------------
# Helper: print DEM metadata
# ---------------------------------------------------------------------------
def print_dem_info(dem_path: Path) -> None:
    """Open the saved DEM and print its metadata."""
    print("\n── DEM Metadata ──────────────────────────────────")
    with rasterio.open(dem_path) as src:
        data = src.read(1, masked=True)
        print(f"  CRS          : {src.crs}")
        print(f"  Shape        : {src.height} rows × {src.width} cols")
        print(f"  Resolution   : {src.res}")
        print(f"  Min elevation: {data.min():.2f} m")
        print(f"  Max elevation: {data.max():.2f} m")
        print(f"  NoData value : {src.nodata}")
    print("──────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Helper: visualise DEM — Plotly interactive (2D heatmap + 3D surface)
# ---------------------------------------------------------------------------
def visualise_dem(dem_path: Path,
                  title: str = "Digital Elevation Model – Bajil Village") -> None:
    """
    Display the DEM using Plotly:
      - Left panel : interactive 2D heatmap with terrain colorscale
      - Right panel: interactive 3D surface for topography exploration
    Opens automatically in the default web browser.
    """
    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        data   = src.read(1).astype(float)
        bounds = src.bounds

    # Replace nodata with NaN for clean rendering
    if nodata is not None:
        data[data == nodata] = np.nan

    # Coordinate axes
    rows, cols = data.shape
    x_vals = np.linspace(bounds.left,  bounds.right, cols)
    y_vals = np.linspace(bounds.top,   bounds.bottom, rows)   # top → bottom

    # ── Build figure with 2 subplots (1 row, 2 cols) ──────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"2D Heatmap – {title}", f"3D Surface – {title}"),
        specs=[[{"type": "heatmap"}, {"type": "surface"}]],
        horizontal_spacing=0.05,
    )

    # ── 2D Heatmap ────────────────────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            z=data,
            x=x_vals,
            y=y_vals,
            colorscale="Earth",
            colorbar=dict(
                title="Elevation (m)",
                titleside="right",
                x=0.45,
            ),
            hovertemplate="Easting: %{x:.0f} m<br>Northing: %{y:.0f} m<br>Elev: %{z:.1f} m<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── 3D Surface ────────────────────────────────────────────────────────
    fig.add_trace(
        go.Surface(
            z=data,
            x=x_vals,
            y=y_vals,
            colorscale="Earth",
            showscale=True,
            colorbar=dict(
                title="Elevation (m)",
                titleside="right",
                x=1.02,
            ),
            hovertemplate="Easting: %{x:.0f}<br>Northing: %{y:.0f}<br>Elev: %{z:.1f} m<extra></extra>",
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
            ),
        ),
        row=1, col=2,
    )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=18)),
        template="plotly_dark",
        height=650,
        width=1300,
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="white", family="Inter, Arial"),
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="Elevation (m)",
            bgcolor="#0f0f1a",
        ),
    )
    fig.update_xaxes(title_text="Easting (m)",  row=1, col=1)
    fig.update_yaxes(title_text="Northing (m)", row=1, col=1)

    # ── Save outputs ──────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build a clean filename from the DEM stem + title
    safe_title = title.replace(" ", "_").replace("–", "-").replace("/", "-")
    stem       = dem_path.stem   # e.g. "dem" or "dem_clipped"
    base_name  = f"{stem}_{safe_title}"

    # 1. Interactive HTML (always available – no extra packages needed)
    html_path = FIGURES_DIR / f"{base_name}.html"
    fig.write_html(str(html_path))
    print(f"  [SAVE] Interactive HTML → {html_path}")

    # 2. Static PNG (requires kaleido; skipped gracefully if not installed)
    png_path = FIGURES_DIR / f"{base_name}.png"
    try:
        fig.write_image(str(png_path), scale=2)
        print(f"  [SAVE] Static PNG      → {png_path}")
    except Exception:
        print("  [WARN] PNG export skipped (install kaleido: pip install kaleido)")

    # ── Show in browser ───────────────────────────────────────────────────
    fig.show()
    print(f"  [INFO] Visualisation opened in browser: {dem_path.name}")


# ---------------------------------------------------------------------------
# Public entry-point (called from main.py)
# ---------------------------------------------------------------------------
def generate_dem_from_contours(
        contour_path: Path = CONTOUR_PATH,
        dem_out_path: Path  = DEM_OUT_PATH,
        resolution: float   = TARGET_RESOLUTION,
) -> Path:
    """
    Full pipeline: contour shapefile → interpolated DEM GeoTIFF.

    Parameters
    ----------
    contour_path : Path
        Input contour shapefile.
    dem_out_path : Path
        Output DEM GeoTIFF path.
    resolution : float
        Pixel size in map units (metres).

    Returns
    -------
    Path
        Path to the saved DEM file.
    """
    print(f"\n{'='*55}")
    print("  DEM Generation from Contour Lines")
    print(f"{'='*55}")

    # -- 1. Load shapefile -------------------------------------------------
    print(f"\n[1/5] Loading contour shapefile: {contour_path.name}")
    if not contour_path.exists():
        sys.exit(f"  [ERROR] File not found: {contour_path}")

    gdf = gpd.read_file(contour_path)
    print(f"  Features : {len(gdf)}")
    print(f"  CRS      : {gdf.crs}")
    print(f"  Columns  : {list(gdf.columns)}")

    # Reproject to a metric CRS if geographic (degrees)
    if gdf.crs is None:
        sys.exit("  [ERROR] Shapefile has no CRS defined.")

    if gdf.crs.is_geographic:
        print("  [WARN] CRS is geographic – reprojecting to UTM …")
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
        print(f"  [INFO] New CRS: {gdf.crs}")

    # -- 2. Detect elevation field ----------------------------------------
    print("\n[2/5] Detecting elevation field …")
    elev_field = detect_elevation_field(gdf)

    # Drop rows with null elevation
    before = len(gdf)
    gdf = gdf.dropna(subset=[elev_field])
    if len(gdf) < before:
        print(f"  [WARN] Dropped {before - len(gdf)} rows with null elevation.")

    # -- 3. Sample (x, y, z) points ---------------------------------------
    print("\n[3/5] Sampling vertex coordinates …")
    x, y, z = sample_xyz_from_contours(gdf, elev_field)
    print(f"  Total sample points: {len(x):,}")
    print(f"  Elevation range    : {z.min():.1f} – {z.max():.1f} m")

    # -- 4. Interpolate ---------------------------------------------------
    print("\n[4/5] Interpolating DEM …")
    bounds = (x.min(), y.min(), x.max(), y.max())
    grid_z, x_min, y_max = interpolate_dem(x, y, z, resolution, bounds)

    # -- 5. Write GeoTIFF -------------------------------------------------
    print("\n[5/5] Writing GeoTIFF …")
    write_dem_tif(grid_z, x_min, y_max, resolution, gdf.crs, dem_out_path)

    # -- Metadata & visualisation -----------------------------------------
    print_dem_info(dem_out_path)
    visualise_dem(dem_out_path)

    return dem_out_path


# ---------------------------------------------------------------------------
# Public: inspect an existing DEM
# ---------------------------------------------------------------------------
def inspect_dem(
        dem_path: Path = DEM_OUT_PATH,
        title: str = "Generated DEM",
) -> None:
    """
    Load a DEM GeoTIFF, print its metadata, and display it with matplotlib.

    Parameters
    ----------
    dem_path : Path
        Path to the DEM raster to inspect.
    title : str
        Plot title to display.
    """
    print(f"\n{'='*55}")
    print(f"  Inspecting DEM: {dem_path.name}")
    print(f"{'='*55}")

    # -- Validate file exists ---------------------------------------------
    if not dem_path.exists():
        raise FileNotFoundError(f"[ERROR] DEM not found: {dem_path}")

    # -- Load and print metadata ------------------------------------------
    with rasterio.open(dem_path) as src:
        nodata  = src.nodata
        data    = src.read(1, masked=True)   # masked array: nodata → masked
        bounds  = src.bounds
        crs     = src.crs
        res     = src.res
        shape   = (src.height, src.width)

    # Mask nodata explicitly if not already masked
    if nodata is not None:
        data = np.ma.masked_equal(data, nodata)

    print(f"\n── DEM Inspection ──────────────────────────────────")
    print(f"  CRS          : {crs}")
    print(f"  Shape        : {shape[0]} rows × {shape[1]} cols")
    print(f"  Resolution   : {res} m")
    print(f"  Min elevation: {data.min():.2f} m")
    print(f"  Max elevation: {data.max():.2f} m")
    print(f"  NoData value : {nodata}")
    print(f"────────────────────────────────────────────────────\n")

    # -- Visualise (Plotly) -----------------------------------------------
    visualise_dem(dem_path, title=title)


# ---------------------------------------------------------------------------
# Public: clip DEM to boundary shapefile
# ---------------------------------------------------------------------------
def clip_dem(
        dem_path: Path      = DEM_OUT_PATH,
        boundary_path: Path = BOUNDARY_PATH,
        out_path: Path      = DEM_CLIPPED_PATH,
) -> Path:
    """
    Clip a DEM raster to the extent and shape of a boundary polygon.

    Steps
    -----
    1. Load boundary shapefile.
    2. If CRS differs from DEM, reproject boundary to DEM CRS.
    3. Clip (mask) the DEM using the boundary geometry.
    4. Save the clipped DEM as a new GeoTIFF.
    5. Visualise the result.

    Parameters
    ----------
    dem_path : Path
        Source DEM raster.
    boundary_path : Path
        Boundary polygon shapefile.
    out_path : Path
        Output path for clipped DEM.

    Returns
    -------
    Path
        Path to the saved clipped DEM.
    """
    print(f"\n{'='*55}")
    print("  Clipping DEM to Boundary")
    print(f"{'='*55}")

    # -- Validate inputs --------------------------------------------------
    if not dem_path.exists():
        raise FileNotFoundError(f"[ERROR] DEM not found: {dem_path}")
    if not boundary_path.exists():
        raise FileNotFoundError(f"[ERROR] Boundary shapefile not found: {boundary_path}")

    # -- Load boundary ----------------------------------------------------
    print(f"\n[1/4] Loading boundary: {boundary_path.name}")
    gdf_boundary = gpd.read_file(boundary_path)
    print(f"  Features : {len(gdf_boundary)}")
    print(f"  CRS      : {gdf_boundary.crs}")

    # -- Open DEM and check CRS -------------------------------------------
    print("\n[2/4] Checking CRS alignment …")
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        print(f"  DEM CRS      : {dem_crs}")
        print(f"  Boundary CRS : {gdf_boundary.crs}")

        # Handle CRS alignment
        if gdf_boundary.crs is None:
            # No CRS defined in shapefile → assume it matches the DEM CRS
            print(f"  [WARN] Boundary has no CRS → assigning DEM CRS: {dem_crs}")
            gdf_boundary = gdf_boundary.set_crs(dem_crs, allow_override=True)
        elif gdf_boundary.crs != dem_crs:
            # CRS defined but differs → reproject to DEM CRS
            print(f"  [WARN] CRS mismatch – reprojecting boundary to DEM CRS …")
            gdf_boundary = gdf_boundary.to_crs(dem_crs)
            print(f"  [INFO] Boundary reprojected to: {dem_crs}")
        else:
            print("  [INFO] CRS match – no reprojection needed.")

        # -- Clip ---------------------------------------------------------
        print("\n[3/4] Clipping DEM …")
        shapes = [geom.__geo_interface__
                  for geom in gdf_boundary.geometry
                  if geom is not None and not geom.is_empty]

        if not shapes:
            raise ValueError("[ERROR] Boundary shapefile contains no valid geometries.")

        clipped_data, clipped_transform = rio_mask(
            src,
            shapes,
            crop=True,
            nodata=src.nodata if src.nodata is not None else -9999,
        )
        clipped_meta = src.meta.copy()

    # -- Update metadata for clipped raster ---------------------------------
    clipped_meta.update({
        "driver"    : "GTiff",
        "height"    : clipped_data.shape[1],
        "width"     : clipped_data.shape[2],
        "transform" : clipped_transform,
        "compress"  : "lzw",
        "nodata"    : clipped_meta.get("nodata", -9999),
    })

    # -- Write output -------------------------------------------------------
    print("\n[4/4] Saving clipped DEM …")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **clipped_meta) as dst:
        dst.write(clipped_data)
    print(f"  [INFO] Clipped DEM saved → {out_path}")

    # -- Inspect + visualise -----------------------------------------------
    inspect_dem(out_path, title="Clipped DEM")

    return out_path


# ===========================================================================
# VALIDATION BLOCK
# ===========================================================================


# ---------------------------------------------------------------------------
# PART 1 — CRS Validation
# ---------------------------------------------------------------------------
def validate_crs(
        dem_path: Path      = DEM_OUT_PATH,
        clipped_path: Path  = DEM_CLIPPED_PATH,
        boundary_path: Path = BOUNDARY_PATH,
) -> dict:
    """
    Load DEM, clipped DEM, and boundary shapefile.
    Print and compare their CRS values and bounding boxes.
    Safely assign CRS to boundary if missing (with extent-based confirmation).

    Returns
    -------
    dict with keys:
        dem_crs, clipped_crs, boundary_crs,
        dem_bounds, boundary_bounds, overlap_ok
    """
    print(f"\n{'='*55}")
    print("  PART 1 — CRS Validation")
    print(f"{'='*55}")

    results = {}

    # ── Load DEM --------------------------------------------------------
    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_bounds = src.bounds
        dem_res    = src.res

    with rasterio.open(clipped_path) as src:
        clipped_crs    = src.crs
        clipped_bounds = src.bounds

    # ── Load boundary ---------------------------------------------------
    gdf = gpd.read_file(boundary_path)
    boundary_crs = gdf.crs

    # ── Print summary ---------------------------------------------------
    print(f"\n  {'Layer':<20} {'CRS':<25} Bounding Box")
    print(f"  {'-'*75}")
    print(f"  {'dem.tif':<20} {str(dem_crs):<25} {dem_bounds}")
    print(f"  {'dem_clipped.tif':<20} {str(clipped_crs):<25} {clipped_bounds}")
    print(f"  {'boundary (raw)':<20} {str(boundary_crs):<25} {gdf.total_bounds.tolist()}")

    # ── Handle missing boundary CRS ------------------------------------
    assigned = False
    if boundary_crs is None:
        print("\n  [WARN] Boundary shapefile has NO CRS defined (.prj missing or blank).")
        bx_min, by_min, bx_max, by_max = gdf.total_bounds

        # ── Smart overlap check (not containment) ─────────────────────────
        # If the extents overlap in both axes, boundary is already in the
        # same projected CRS as the DEM → assign directly with set_crs().
        # Otherwise fall back to geographic assumption.
        extents_overlap_x = (bx_min < dem_bounds.right) and (bx_max > dem_bounds.left)
        extents_overlap_y = (by_min < dem_bounds.top)   and (by_max > dem_bounds.bottom)

        if extents_overlap_x and extents_overlap_y:
            print(f"  [AUTO] Extents overlap DEM → boundary is already in {dem_crs}.")
            print(f"         Assigning CRS directly (set_crs, no reprojection).")
            gdf = gdf.set_crs(dem_crs, allow_override=True)
        else:
            print("  [WARN] Boundary extents do NOT overlap DEM — trying EPSG:4326 → reproject …")
            gdf = gdf.set_crs("EPSG:4326").to_crs(dem_crs)
        boundary_crs = gdf.crs
        assigned = True

    elif boundary_crs != dem_crs:
        print(f"\n  [WARN] CRS mismatch! Reprojecting boundary {boundary_crs} → {dem_crs} …")
        gdf = gdf.to_crs(dem_crs)
        boundary_crs = gdf.crs

    else:
        print("\n  [OK] All CRS values match.")

    if assigned:
        print(f"  [INFO] Final boundary CRS: {boundary_crs}")

    # ── Overlap check --------------------------------------------------
    bx_min, by_min, bx_max, by_max = gdf.total_bounds
    overlap_x = (bx_min < dem_bounds.right) and (bx_max > dem_bounds.left)
    overlap_y = (by_min < dem_bounds.top)   and (by_max > dem_bounds.bottom)
    overlap_ok = overlap_x and overlap_y

    if overlap_ok:
        print("  [OK] Boundary OVERLAPS the DEM extent correctly. ✓")
    else:
        print("  [ERROR] Boundary does NOT overlap DEM! Clipping will produce empty result.")

    results.update({
        "dem_crs"       : dem_crs,
        "clipped_crs"   : clipped_crs,
        "boundary_crs"  : boundary_crs,
        "dem_bounds"    : dem_bounds,
        "boundary_bounds": gdf.total_bounds,
        "overlap_ok"    : overlap_ok,
        "gdf_aligned"   : gdf,       # reprojected GeoDataFrame for reuse
    })
    return results


# ---------------------------------------------------------------------------
# PART 2 — DEM Quality Check
# ---------------------------------------------------------------------------
def check_dem_quality(dem_path: Path = DEM_OUT_PATH) -> dict:
    """
    Compute descriptive statistics on the DEM and detect anomalies:
      - elevation range too narrow (flat terrain)
      - extreme spikes (z-score > 5)
      - nodata proportion

    Returns
    -------
    dict with keys: min, max, mean, std, nodata_pct, warnings
    """
    print(f"\n{'='*55}")
    print("  PART 2 — DEM Quality Check")
    print(f"{'='*55}")

    with rasterio.open(dem_path) as src:
        nodata  = src.nodata
        raw     = src.read(1).astype(float)
        res     = src.res
        shape   = (src.height, src.width)

    # Mask nodata
    if nodata is not None:
        valid = raw[raw != nodata]
    else:
        valid = raw.ravel()

    total_cells = raw.size
    valid_cells = valid.size
    nodata_pct  = 100.0 * (total_cells - valid_cells) / total_cells

    z_min  = float(valid.min())
    z_max  = float(valid.max())
    z_mean = float(valid.mean())
    z_std  = float(valid.std())
    z_range = z_max - z_min

    print(f"\n  ── Elevation Statistics ─────────────────────────────")
    print(f"  Min elevation   : {z_min:.2f} m")
    print(f"  Max elevation   : {z_max:.2f} m")
    print(f"  Range           : {z_range:.2f} m")
    print(f"  Mean            : {z_mean:.2f} m")
    print(f"  Std deviation   : {z_std:.2f} m")
    print(f"  NoData cells    : {nodata_pct:.1f}% of total")
    print(f"  Resolution      : {res[0]} × {res[1]} m")
    print(f"  Shape           : {shape[0]} rows × {shape[1]} cols")

    aspect_ratio = shape[0] / shape[1]
    print(f"  Aspect ratio    : {aspect_ratio:.2f} (rows/cols)")

    # ── Warnings --------------------------------------------------------
    warnings_list = []

    if z_range < 5:
        msg = f"[WARN] Elevation range is very small ({z_range:.1f} m) — possible flat/bad DEM."
        warnings_list.append(msg)
        print(f"  {msg}")

    if z_std < 1:
        msg = f"[WARN] Std deviation is extremely low ({z_std:.2f} m) — DEM may be uniform."
        warnings_list.append(msg)
        print(f"  {msg}")

    if nodata_pct > 20:
        msg = f"[WARN] High NoData ratio ({nodata_pct:.1f}%) — check clipping or interpolation."
        warnings_list.append(msg)
        print(f"  {msg}")

    # Spike detection via z-score (values > 5σ from mean)
    z_scores = np.abs((valid - z_mean) / (z_std + 1e-9))
    spike_count = int((z_scores > 5).sum())
    if spike_count > 0:
        msg = f"[WARN] {spike_count} spike pixels detected (|z-score| > 5)."
        warnings_list.append(msg)
        print(f"  {msg}")

    if aspect_ratio > 5 or aspect_ratio < 0.2:
        msg = f"[WARN] Strange aspect ratio ({aspect_ratio:.2f}). Check DEM extent."
        warnings_list.append(msg)
        print(f"  {msg}")

    if not warnings_list:
        print("  [OK] No quality issues detected. DEM looks scientifically valid. ✓")

    return {
        "min": z_min, "max": z_max, "mean": z_mean,
        "std": z_std, "range": z_range,
        "nodata_pct": nodata_pct, "warnings": warnings_list,
    }


# ---------------------------------------------------------------------------
# PART 3 — matplotlib validation plot: DEM only
# ---------------------------------------------------------------------------
def plot_dem(
        dem_path: Path = DEM_OUT_PATH,
        save_name: str = "dem_check.png",
) -> None:
    """
    Professional matplotlib validation plot of a DEM raster.
    Saved to outputs/figures/<save_name>.
    """
    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        data   = src.read(1).astype(float)
        bounds = src.bounds
        crs    = src.crs
        res    = src.res

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(
        data,
        cmap="terrain",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        interpolation="bilinear",
        vmin=np.nanmin(data),
        vmax=np.nanmax(data),
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Elevation (m)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(f"DEM Validation | CRS: {crs} | Res: {res[0]}m",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m)",  color="white")
    ax.set_ylabel("Northing (m)", color="white")
    ax.tick_params(colors="white")
    ax.ticklabel_format(style="plain", axis="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Stats annotation
    valid = data[~np.isnan(data)]
    stats_txt = (f"Min: {np.nanmin(data):.1f} m   Max: {np.nanmax(data):.1f} m\n"
                 f"Mean: {np.nanmean(data):.1f} m   Std: {np.nanstd(data):.1f} m")
    ax.text(0.02, 0.04, stats_txt, transform=ax.transAxes,
            fontsize=9, color="white", verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f0f1a", alpha=0.8))

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGURES_DIR / save_name
    plt.savefig(str(out_png), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [SAVE] {save_name} → {out_png}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# PART 3b — matplotlib validation plot: DEM + boundary overlay
# ---------------------------------------------------------------------------
def check_alignment(
        dem_path: Path      = DEM_OUT_PATH,
        boundary_path: Path = BOUNDARY_PATH,
        dem_crs: "CRS | None" = None,
        save_name: str      = "dem_boundary_check.png",
) -> None:
    """
    Overlay the boundary shapefile on the DEM to visually confirm alignment.
    If boundary has no CRS, it is assigned dem_crs before plotting.
    Saved to outputs/figures/<save_name>.
    """
    with rasterio.open(dem_path) as src:
        nodata    = src.nodata
        data      = src.read(1).astype(float)
        bounds    = src.bounds
        _dem_crs  = src.crs

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    # Resolve CRS
    target_crs = dem_crs if dem_crs is not None else _dem_crs

    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs, allow_override=True)
    elif gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # DEM background
    im = ax.imshow(
        data,
        cmap="terrain",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        interpolation="bilinear",
        vmin=np.nanmin(data),
        vmax=np.nanmax(data),
        alpha=0.85,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Elevation (m)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Boundary overlay
    gdf.boundary.plot(
        ax=ax,
        color="#ff4d4d",
        linewidth=2.5,
        linestyle="--",
        label="Study Boundary",
    )

    legend_patch = mpatches.Patch(edgecolor="#ff4d4d", facecolor="none",
                                  linestyle="--", linewidth=2, label="Study Boundary")
    ax.legend(handles=[legend_patch], loc="upper right",
              facecolor="#0f0f1a", edgecolor="#444", labelcolor="white")

    ax.set_title("DEM + Boundary Alignment", color="white",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m)",  color="white")
    ax.set_ylabel("Northing (m)", color="white")
    ax.tick_params(colors="white")
    ax.ticklabel_format(style="plain", axis="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGURES_DIR / save_name
    plt.savefig(str(out_png), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  [SAVE] {save_name} → {out_png}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master orchestrator: run full validation suite
# ---------------------------------------------------------------------------
def run_validation(
        dem_path: Path      = DEM_OUT_PATH,
        clipped_path: Path  = DEM_CLIPPED_PATH,
        boundary_path: Path = BOUNDARY_PATH,
) -> None:
    """
    Run the complete DEM validation suite in order:
        1. CRS validation
        2. Quality check
        3. matplotlib DEM plot
        4. matplotlib DEM + boundary alignment plot
    If CRS was fixed in step 1, the corrected GeoDataFrame is passed to step 4.
    """
    print(f"\n{'#'*55}")
    print("  DEM VALIDATION SUITE")
    print(f"{'#'*55}")

    # 1. CRS
    crs_results = validate_crs(dem_path, clipped_path, boundary_path)

    # 2. Quality
    check_dem_quality(dem_path)

    # 3. DEM plot (matplotlib)
    print(f"\n{'='*55}")
    print("  PART 3 — Matplotlib Validation Plots")
    print(f"{'='*55}")
    plot_dem(dem_path, save_name="dem_check.png")

    # 4. Alignment plot (pass the corrected dem_crs)
    check_alignment(
        dem_path      = dem_path,
        boundary_path = boundary_path,
        dem_crs       = crs_results["dem_crs"],
        save_name     = "dem_boundary_check.png",
    )

    print(f"\n{'#'*55}")
    print("  VALIDATION COMPLETE")
    if not crs_results["overlap_ok"]:
        print("  [WARNING] Boundary does NOT overlap DEM!")
    else:
        print("  All checks passed. DEM is ready for hydrology analysis. ✓")
    print(f"{'#'*55}\n")


# ---------------------------------------------------------------------------
# NEW: Clip Vector Boundary to Raster Extent
# ---------------------------------------------------------------------------
def clip_vector_boundary_to_raster(
        dem_path: Path      = DEM_OUT_PATH,
        boundary_path: Path = BOUNDARY_PATH,
        out_path: Path      = BOUNDARY_CLIPPED_PATH
) -> Path:
    """
    Clips a vector boundary shapefile to match the exact physical extent of a DEM.

    Pipeline:
    1. Open DEM to get bounds and CRS.
    2. Create a clipping Polygon from DEM bounds.
    3. Load boundary vector and align CRS.
    4. Perform spatial clipping.
    5. Save the resulting vector.
    """
    print(f"\n{'='*55}")
    print("  Clipping Vector Boundary to DEM Extent")
    print(f"{'='*55}")

    # -- 1. Extract DEM Metadata -------------------------------------------
    with rasterio.open(dem_path) as src:
        dem_bounds = src.bounds
        dem_crs    = src.crs
        print(f"[1/4] DEM Bounds: {dem_bounds}")
        print(f"      DEM CRS   : {dem_crs}")

    # -- 2. Create Bounding Box Polygon ------------------------------------
    # box(minx, miny, maxx, maxy)
    dem_bbox_geom = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
    gdf_dem_extent = gpd.GeoDataFrame({'geometry': [dem_bbox_geom]}, crs=dem_crs)

    # -- 3. Load & Align Boundary ------------------------------------------
    print(f"\n[2/4] Loading boundary: {boundary_path.name}")
    gdf_boundary = gpd.read_file(boundary_path)
    print(f"      Boundary Bounds (Pre-clip) : {gdf_boundary.total_bounds.tolist()}")

    # Assign CRS if missing (based on previous validation logic)
    if gdf_boundary.crs is None:
        print(f"      [WARN] Boundary has no CRS. Assigning DEM CRS: {dem_crs}")
        gdf_boundary.set_crs(dem_crs, inplace=True, allow_override=True)
    elif gdf_boundary.crs != dem_crs:
        print(f"      [INFO] Reprojecting Boundary to match DEM CRS...")
        gdf_boundary = gdf_boundary.to_crs(dem_crs)

    # -- 4. Perform Clip ---------------------------------------------------
    print("\n[3/4] Performing Vector Clip...")
    # Using gpd.clip for clear intersection with the DEM footprint
    gdf_clipped = gpd.clip(gdf_boundary, gdf_dem_extent)

    if gdf_clipped.empty:
        print("      [ERROR] Clipping resulted in an empty geometry. No overlap!")
        return None

    # -- 5. Save and Report ------------------------------------------------
    print(f"\n[4/4] Finalizing Output...")
    print(f"      Boundary Bounds (Post-clip): {gdf_clipped.total_bounds.tolist()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_clipped.to_file(out_path)

    print(f"\n✅ Success: Clipped boundary saved to:")
    print(f"   → {out_path}")

    return out_path

