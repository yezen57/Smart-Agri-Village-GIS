"""
preprocess.py
=============
Generates a DEM raster from contour lines (vector shapefile).

Pipeline:
    1. Load contour shapefile with geopandas
    2. Auto-detect the elevation attribute field
    3. Sample (x, y, z) points from contour geometries
    4. Interpolate a regular grid using scipy's griddata (linear method)
    5. Write the result as a GeoTIFF (data/processed/dem.tif)
    6. Print DEM metadata (CRS, shape, min/max elevation)
    7. Visualise the DEM with matplotlib
"""

import sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # …/project/
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
CONTOUR_PATH = RAW_DIR / "contor.shp"
DEM_OUT_PATH = PROC_DIR / "dem.tif"

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
# Helper: visualise DEM
# ---------------------------------------------------------------------------
def visualise_dem(dem_path: Path) -> None:
    """
    Display the DEM using matplotlib with a terrain colourmap and a colourbar.
    """
    with rasterio.open(dem_path) as src:
        data  = src.read(1, masked=True)
        bounds = src.bounds

    fig, ax = plt.subplots(figsize=(10, 8))

    img = ax.imshow(
        data,
        cmap="terrain",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        interpolation="bilinear",
    )

    cbar = fig.colorbar(img, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Elevation (m)", fontsize=11)

    ax.set_title("Digital Elevation Model – Bajil Village", fontsize=14, fontweight="bold")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain", axis="both")

    plt.tight_layout()
    plt.show()


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
