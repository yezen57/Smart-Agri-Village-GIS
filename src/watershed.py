"""
watershed.py
============
Phase 4: Flow Analysis and Stream Network Extraction.
Computes Flow Direction, Accumulation, and vectorizes Channels.
"""

import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape, LineString
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pysheds.grid import Grid
from pathlib import Path

# Paths configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
OUT_DIR      = PROJECT_ROOT / "outputs"
FIG_DIR      = OUT_DIR / "figures"

# Inputs from Phase 3
DEM_FILLED    = OUT_DIR / "dem_filled.tif"

# Outputs for Phase 4
FLOW_DIR_TIF  = OUT_DIR / "flow_dir.tif"
FLOW_ACC_TIF  = OUT_DIR / "flow_acc.tif"
CHANNELS_SHP  = OUT_DIR / "channels.shp"


def run_flow_analysis(dem_path: Path = DEM_FILLED, threshold_pct: float = 0.5) -> None:
    """
    Main Orchestrator for Phase 4:
    1. Compute Flow Direction (D8).
    2. Compute Flow Accumulation.
    3. Extract Stream Network.
    4. Save Vector Channels.
    """
    print(f"\n{'='*55}")
    print("  PHASE 4: WATERSHED & FLOW ANALYSIS (PYSHEDS)")
    print(f"{'='*55}")

    if not dem_path.exists():
        print(f"  [ERROR] Input Filled DEM not found: {dem_path}")
        return

    # -- [1] Initialize Grid & Load DEM -------------------------------------
    print(f"\n[1/7] Initializing Grid and Loading DEM …")
    grid = Grid.from_raster(str(dem_path))
    dem  = grid.read_raster(str(dem_path))
    print(f"      Shape: {dem.shape} | Bounds: {grid.extent}")

    # -- [2] Hydrological Conditioning (Crucial for Clarity) ----------------
    print("\n[2/7] Conditioning Grid: Filling Pits & Depressions (Removing Noise) …")
    # 1. Fill pits (small artifacts)
    pit_filled_dem = grid.fill_pits(dem)
    # 2. Fill depressions (larger sinks)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    # 3. Resolve flats (ensures continuous flow paths)
    inflated_dem = grid.resolve_flats(flooded_dem)

    # -- [3] Compute Flow Direction (D8) ------------------------------------
    print("\n[3/7] Computing Flow Direction (D8) on Conditioned DEM …")
    # Using the conditioned DEM ensures connected stream networks without noise
    fdir = grid.flowdir(inflated_dem)
    
    # Save Flow Direction
    grid.to_raster(fdir, str(FLOW_DIR_TIF))
    print(f"      [INFO] Flow Direction saved → {FLOW_DIR_TIF.name}")

    # -- [3] Compute Flow Accumulation --------------------------------------
    print("\n[3/7] Computing Flow Accumulation …")
    acc = grid.accumulation(fdir)
    
    # Save Flow Accumulation
    grid.to_raster(acc, str(FLOW_ACC_TIF))
    print(f"      [INFO] Flow Accumulation saved → {FLOW_ACC_TIF.name}")

    # -- [4] Stream Extraction Strategy -------------------------------------
    print("\n[4/7] Selecting Accumulation Threshold …")
    max_acc = acc.max()
    # threshold based on percentage of max accumulation or fixed value
    threshold = (threshold_pct / 100.0) * max_acc
    if threshold < 100: threshold = 100 # Safety floor
    
    print(f"      Max Accumulation: {max_acc:.0f} cells")
    print(f"      Selected Threshold ({threshold_pct}%): {threshold:.0f} cells")

    # -- [5] Extract Channels (Vectorization) -------------------------------
    print("\n[5/7] Extracting Stream Network (Vectorizing) …")
    # pysheds has a built-in river network extractor that returns a GeoDataFrame-ready structure
    branches = grid.extract_river_network(fdir, acc > threshold)
    
    # Convert to GeoDataFrame
    features_list = []
    for branch in branches['features']:
        line = shape(branch['geometry'])
        features_list.append({'geometry': line, 'id': len(features_list)})
    
    gdf_channels = gpd.GeoDataFrame(features_list, crs=grid.crs)
    
    # Save Shapefile
    gdf_channels.to_file(CHANNELS_SHP)
    print(f"      [INFO] Channels Shapefile saved → {CHANNELS_SHP.name}")

    # -- [6] Visualization --------------------------------------------------
    print("\n[6/7] Generating Visualizations (Matplotlib & Plotly) …")
    # Static plots
    plot_watershed_results(acc, gdf_channels, dem)
    # Interactive plots
    plot_watershed_interactive(acc, gdf_channels, dem, grid) # تم تمرير المتغير المفقود هنا

    # -- [7] Engineering Validation -----------------------------------------
    print(f"\n✔ Flow Direction computed")
    print(f"✔ Flow Accumulation computed")
    print(f"✔ Streams extracted (Threshold: {threshold:.0f})")
    print(f"✔ Channels shapefile created: {CHANNELS_SHP.name}")
    print(f"\nFinal: Ready for Sensitivity Analysis stage.\n")


def run_sensitivity_analysis(dem_path: Path = DEM_FILLED):
    """
    Performs sensitivity analysis on 3 thresholds: 50, 100, 300.
    Calculates stats and generates comparative plots.
    """
    print(f"\n{'='*55}")
    print("  PHASE 4.1: THRESHOLD SENSITIVITY ANALYSIS")
    print(f"{'='*55}")

    grid = Grid.from_raster(str(dem_path))
    dem  = grid.read_raster(str(dem_path))
    
    # 1. Conditioning & Flow Dir
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)
    fdir = grid.flowdir(inflated_dem)
    acc  = grid.accumulation(fdir)

    thresholds = [50, 100, 300]
    results = []

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    fig.patch.set_facecolor('#fdfdfd')

    for i, threshold in enumerate(thresholds):
        print(f"\n[Testing] Threshold: {threshold} cells")
        
        # Extract network
        branches = grid.extract_river_network(fdir, acc > threshold)
        
        features_list = []
        total_length = 0
        for branch in branches['features']:
            line = shape(branch['geometry'])
            features_list.append({'geometry': line, 'id': len(features_list)})
            total_length += line.length # Unit is map units (meters)

        gdf = gpd.GeoDataFrame(features_list, crs=grid.crs)
        
        # Save specific shapefile
        out_shp = OUT_DIR / f"channels_{threshold}.shp"
        gdf.to_file(out_shp)
        
        # Print stats
        seg_count = len(gdf)
        print(f"      Count: {seg_count} segments | Total Length: {total_length:.2f} m")
        
        results.append({
            'threshold': threshold,
            'count': seg_count,
            'length': total_length,
            'gdf': gdf
        })

        # Plot over DEM
        axes[i].imshow(dem, cmap='terrain', alpha=0.6)
        gdf.plot(ax=axes[i], color='blue', linewidth=1.0)
        axes[i].set_title(f"Threshold: {threshold}\n({seg_count} segs, {total_length/1000:.1f} km)", fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_sensitivity_comparison.png", dpi=200)
    print(f"\n[SAVE] Sensitivity Comparison Plot → figures/threshold_sensitivity_comparison.png")
    
    return results



def plot_watershed_results(acc, gdf_channels, dem):
    """Visual QA for watershed results."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#f4f4f8')

    # Plot 1: Flow Accumulation (Log Scale)
    print("      Plotting Accumulation Map …")
    # Mask zeros for log plot
    acc_masked = np.where(acc <= 0, 1, acc) 
    im1 = ax[0].imshow(acc_masked, extent=None, cmap='Blues', norm=LogNorm())
    plt.colorbar(im1, ax=ax[0]).set_label('Accumulated Cells (Log Scale)')
    ax[0].set_title('Flow Accumulation Network', fontweight='bold')
    ax[0].axis('off')

    # Plot 2: DEM with Channels Overlay
    print("      Plotting Stream Network Overlay …")
    ax[1].imshow(dem, extent=None, cmap='terrain', alpha=0.7)
    gdf_channels.plot(ax=ax[1], color='blue', linewidth=1.2, label='Streams')
    ax[1].set_title('Extracted Channels on Terrain', fontweight='bold')
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "watershed_flow_analysis.png", dpi=200)
    
    # Independent Accumulation plot as requested
    plt.figure(figsize=(10, 8))
    plt.imshow(acc_masked, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Flow Accumulation')
    plt.title('Watershed Drainage Density')
    plt.savefig(FIG_DIR / "flow_accumulation.png", dpi=150)
    plt.close()

    print(f"      [SAVE] QA Plots → {FIG_DIR.name}")


def plot_watershed_interactive(acc, gdf_channels, dem, grid):
    """Generates interactive Plotly HTML reports for Watershed Analysis."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Interactive 2D Accumulation Map
    print("      Creating Interactive 2D Accumulation Map …")
    # Log scale for accumulation visibility
    acc_log = np.log10(np.where(acc <= 0, 1, acc))
    
    fig2d = go.Figure(data=go.Heatmap(
        z=acc_log,
        colorscale='Hot',  # ألوان أكثر تبايناً للأودية
        colorbar=dict(title="Flow Intensity (Log10)")
    ))
    fig2d.update_layout(
        title="Interactive Flow Accumulation Network - High Contrast",
        template="plotly_dark"
    )
    
    # Save HTML and PNG
    html_2d = FIG_DIR / "interactive_flow_accumulation.html"
    png_2d  = FIG_DIR / "interactive_flow_accumulation.png"
    fig2d.write_html(str(html_2d))
    fig2d.write_image(str(png_2d), scale=2) # يحتاج kaleido
    
    # 2. Interactive 3D Terrain with Streams Overlay
    print("      Creating Enhanced Interactive 3D Terrain + Streams …")
    
    z_data = np.asarray(dem)
    
    fig3d = go.Figure()
    
    # Add Terrain Surface
    fig3d.add_trace(go.Surface(
        z=z_data, 
        colorscale='Greens', 
        opacity=0.9,
        showscale=False,
        name='Terrain'
    ))
    
    # -- Add Streams as 3D Lines --------------------------------------------
    # we need to map map-coordinates to grid-indices correctly for overlay
    # For a visualization, we'll plot a subset of major channels
    z_offset = 2.0 # رفع الخط قليلاً فوق الأرض ليظهر بوضوح
    
    for _, row in gdf_channels.iterrows():
        coords = np.array(row.geometry.coords)
        if len(coords) < 2: continue
        
        # Convert map coordinates (X, Y) to grid indices (Col, Row)
        # Using pysheds grid transform (Tilde ~ is used for inverse)
        cols, rows = ~grid.affine * (coords[:, 0], coords[:, 1])
        
        # Plotly surface uses indices directly [0..height, 0..width]
        # Important: Row index in Surface plot corresponds to first dimension
        fig3d.add_trace(go.Scatter3d(
            x=cols,
            y=rows,
            z=z_data[rows.astype(int), cols.astype(int)] + z_offset,
            mode='lines',
            line=dict(color='cyan', width=4),
            name='Stream',
            showlegend=False
        ))

    # Finalize 3D layout
    fig3d.update_layout(
        title="Interactive 3D Hydraulic Model - Streams Overlay",
        scene=dict(
            xaxis=dict(title="West-East"),
            yaxis=dict(title="North-South"),
            zaxis=dict(title="Elevation (m)"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=3, z=0.5)
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    html_3d = FIG_DIR / "interactive_3d_watershed.html"
    png_3d  = FIG_DIR / "interactive_3d_watershed.png"
    fig3d.write_html(str(html_3d))
    fig3d.write_image(str(png_3d), scale=2)
    print(f"      [SAVE] Interactive HTMLs and PNGs saved to {FIG_DIR.name}")



if __name__ == "__main__":
    run_flow_analysis()
