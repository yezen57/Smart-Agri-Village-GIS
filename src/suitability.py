"""
suitability.py (MCDA Edition)
============================
Phase 5: Scientific Multi-Criteria Decision Analysis (MCDA).
Utilizes Fuzzy Logic, Hard Constraints, Gaussian Smoothing, and DBSCAN Clustering.
"""

import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import distance_transform_edt, gaussian_filter
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, shape, MultiPolygon
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
FIG_DIR      = OUT_DIR / "figures"

# Inputs
DEM_TIF       = OUT_DIR / "dem_filled.tif"
SLOPE_TIF     = OUT_DIR / "slope.tif"
FLOW_ACC_TIF  = OUT_DIR / "flow_acc.tif"
ROADS_SHP     = PROC_DIR / "roads_clipped.shp"

# Outputs
SUITABILITY_TIF = OUT_DIR / "suitability_fuzzy.tif"
ZONES_SHP       = OUT_DIR / "suitability_zones.shp"
SITES_SHP       = OUT_DIR / "top_sites_points.shp"


def fuzzy_slope(slope):
    """Fuzzy membership for slope: 0-5 (1.0), 5-10 (0.7), 10-15 (0.3), >15 (0.0)."""
    fuzzy = np.zeros_like(slope, dtype=float)
    fuzzy[slope <= 5] = 1.0
    mask_mid1 = (slope > 5) & (slope <= 10)
    fuzzy[mask_mid1] = 0.7
    mask_mid2 = (slope > 10) & (slope <= 15)
    fuzzy[mask_mid2] = 0.3
    fuzzy[slope > 15] = 0.0
    return fuzzy


def fuzzy_roads(dist):
    """Fuzzy membership for roads: 20-200m (1.0), <20m (safety penalty), >1000m (penalty)."""
    fuzzy = np.zeros_like(dist, dtype=float)
    fuzzy[(dist >= 20) & (dist <= 200)] = 1.0
    fuzzy[dist < 20] = 0.4 # Penalty for being too close (dust, noise, safety)
    fuzzy[dist > 200] = np.maximum(0.2, 1.0 - (dist[dist > 200] - 200) / 1000.0)
    return fuzzy


def run_suitability_analysis(weights: dict = None) -> None:
    if weights is None:
        weights = {"slope": 0.4, "flood": 0.4, "road": 0.2}

    print(f"\n{'='*55}")
    print("  PHASE 5: SCIENTIFIC MCDA SITE SELECTION")
    print(f"{'='*55}")

    # 1. Loading and Preparation
    print("\n[1/8] Loading layers and aligning CRS …")
    with rasterio.open(DEM_TIF) as src:
        dem = src.read(1).astype(float)
        meta = src.meta.copy()
        affine = src.transform
        crs = src.crs

    with rasterio.open(SLOPE_TIF) as src:
        slope = src.read(1).astype(float)
    
    with rasterio.open(FLOW_ACC_TIF) as src:
        flow_acc = src.read(1).astype(float)

    # 2. Distance to Roads
    roads_gdf = gpd.read_file(ROADS_SHP)
    road_shapes = ((geom, 1) for geom in roads_gdf.geometry)
    road_mask = features.rasterize(road_shapes, out_shape=slope.shape, transform=affine, fill=0)
    dist_to_roads = distance_transform_edt(road_mask == 0) * affine[0]

    # 3. Fuzzy Normalization
    print("[2/8] Applying Fuzzy Membership Functions …")
    s_fuzzy = fuzzy_slope(slope)
    r_fuzzy = fuzzy_roads(dist_to_roads)
    # Flood fuzzy: Inverse log normalization
    f_log = np.log1p(flow_acc)
    f_fuzzy = 1.0 - (f_log - f_log.min()) / (f_log.max() - f_log.min() + 1e-6)

    # 4. Hard Constraints
    print("[3/8] Applying Hard Engineering Constraints …")
    mask = (slope > 15) | (flow_acc > 1000) # Exclude steep slopes and major wadis
    
    # 5. Weighted MCDA Model
    print("[4/8] Computing Weighted MCDA (Slope 0.4, Flood 0.4, Road 0.2) …")
    suitability = (s_fuzzy * weights['slope'] + 
                   f_fuzzy * weights['flood'] + 
                   r_fuzzy * weights['road'])
    
    suitability[mask] = 0.0 # Apply constraints

    # 6. Post Processing (Smoothing & Clipping)
    print("[5/8] Post-Processing: Gaussian Smoothing & Calibration …")
    suitability = gaussian_filter(suitability, sigma=0.5)
    suitability = (suitability - suitability.min()) / (suitability.max() - suitability.min() + 1e-6)
    suitability = np.clip(suitability, 0.0, 0.97)

    # Save Raster
    meta.update(dtype='float32', nodata=-9999)
    with rasterio.open(SUITABILITY_TIF, 'w', **meta) as dst:
        dst.write(suitability.astype('float32'), 1)

    # 7. Zonal Clustering (DBSCAN)
    print("[6/8] Performing Zonal Clustering (DBSCAN) on Top 10% …")
    threshold = np.percentile(suitability, 90)
    y_idx, x_idx = np.where(suitability >= threshold)
    
    if len(x_idx) > 0:
        coords = np.column_stack((x_idx, y_idx))
        db = DBSCAN(eps=2, min_samples=5).fit(coords)
        labels = db.labels_
        
        zones_geom = []
        point_features = []
        
        for cluster_id in set(labels):
            if cluster_id == -1: continue
            
            cluster_mask = labels == cluster_id
            cluster_coords = coords[cluster_mask]
            
            # Simple polygonization of cluster
            # For brevity, we create a poly-buffer around points
            points_geom = [Point(rasterio.transform.xy(affine, r, c)) for c, r in cluster_coords]
            gdf_cluster = gpd.GeoDataFrame(geometry=points_geom, crs=crs)
            poly = gdf_cluster.buffer(affine[0]*0.7).unary_union
            
            if not poly.is_empty:
                zones_geom.append({'geometry': poly, 'cluster_id': int(cluster_id)})
                # Centroid for site points
                centroid = poly.centroid
                point_features.append({'geometry': centroid, 'cluster_id': int(cluster_id)})

        gdf_zones = gpd.GeoDataFrame(zones_geom, crs=crs)
        gdf_zones.to_file(ZONES_SHP)
        
        gdf_sites = gpd.GeoDataFrame(point_features, crs=crs)
        gdf_sites.to_file(SITES_SHP)
        print(f"      ✔ {len(gdf_zones)} Suitability Zones detected.")
    else:
        print("      [WARN] No high suitability areas found.")

    # 8. Visualization
    print("[7/8] Generating Scientific & 3D Visualization …")
    plot_scientific_summary(suitability, gdf_zones, roads_gdf)
    plot_suitability_3d(suitability, gdf_sites, dem)
    
    # 9. Conclusion
    print(f"\n{'='*55}")
    print(f"MCDA ANALYSIS COMPLETE")
    print(f"Max Index: {np.max(suitability):.3f}")
    print(f"Valid Zones: {len(gdf_zones)}")
    print(f"{'='*55}\n")

def plot_scientific_summary(suitability, gdf_zones, roads_gdf):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(suitability, cmap='viridis', origin='upper')
    plt.colorbar(im, label='MCDA Suitability Index (Fuzzy)')
    
    roads_gdf.plot(ax=ax, color='red', alpha=0.5, linewidth=0.5, label='Road Infrastructure')
    if not gdf_zones.empty:
        gdf_zones.boundary.plot(ax=ax, color='white', linewidth=1.5, label='High Suitability Zones')
    
    ax.set_title("MCDA Suitability Model (Fuzzy-Gaussian-DBSCAN)", fontweight='bold')
    plt.legend()
    plt.savefig(FIG_DIR / "suitability_mcda_scientific.png", dpi=250)
    plt.close()
    print(f"      [SAVE] Scientific Map → figures/suitability_mcda_scientific.png")


def plot_suitability_3d(suitability, gdf_sites, dem):
    """WOW Factor: 3D Terrain overlaid with Suitability Heatmap and Top Sites."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Create the Surface Trace
    # We use 'dem' for Z (height) and 'suitability' for surfacecolor
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=dem,
        surfacecolor=suitability,
        colorscale='RdYlGn',
        name='Suitability Terrain',
        colorbar=dict(title="Suitability Index", x=0.9)
    ))

    # 2. Add Top Sites as 3D Points
    # We need to map grid indices (r, c) to elevation for correct Z placement
    # Using the points from gdf_sites (already in project CRS)
    # To keep it simple in array space, we use row/col indices for Scatter3d
    flat_indices = np.argsort(suitability.ravel())[::-1][:70]
    r_idx, c_idx = np.unravel_index(flat_indices, suitability.shape)
    z_idx = dem[r_idx, c_idx] + 2 # Offset slightly above terrain

    fig.add_trace(go.Scatter3d(
        x=c_idx,
        y=r_idx,
        z=z_idx,
        mode='markers',
        marker=dict(size=5, color='gold', symbol='diamond', line=dict(width=1, color='white')),
        name='Top 70 Suggested Sites',
        hovertext=[f"Suitability: {s:.2f}" for s in suitability[r_idx, c_idx]]
    ))

    fig.update_layout(
        title="3D Suitability Intelligence - Smart Village Site Selection",
        template="plotly_dark",
        scene=dict(
            xaxis_title="Grid X",
            yaxis_title="Grid Y",
            zaxis_title="Elevation (m)",
            aspectratio=dict(x=1, y=2, z=0.3) # Vertical exaggeration for clarity
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=900
    )

    html_path = FIG_DIR / "interactive_3d_suitability_model.html"
    fig.write_html(str(html_path))
    print(f"      [SAVE] 3D Interactive Model → figures/interactive_3d_suitability_model.html")


if __name__ == "__main__":
    run_suitability_analysis()
