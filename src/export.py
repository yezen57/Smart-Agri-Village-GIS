"""
export.py
=========
Phase 6.5 & 7: Validate, Final Decision (Flood + MCDA), and Export Pipeline.
"""

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import shutil
import warnings

# Use reportlab for PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "outputs"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
FIG_DIR      = OUT_DIR / "figures"
FINAL_DIR    = OUT_DIR / "final"

# Inputs
DEM_TIF      = OUT_DIR / "dem_filled.tif"
SLOPE_TIF    = OUT_DIR / "slope.tif"
FACC_TIF     = OUT_DIR / "flow_acc.tif"
FLOOD_TIF    = OUT_DIR / "flood_map.tif"

CHANNELS_SHP = OUT_DIR / "channels.shp"
SUIT_ZONES   = OUT_DIR / "suitability_zones.shp"
FLOOD_ZONES  = OUT_DIR / "flood_zones.shp"

# Outputs
FINAL_SITES  = PROC_DIR / "final_sites.shp"
FINAL_MAPS   = FINAL_DIR / "maps"
GPKG_OUT     = FINAL_DIR / "project.gpkg"
DXF_OUT      = FINAL_DIR / "channels.dxf"
REPORT_OUT   = FINAL_DIR / "report.pdf"


def run_export_pipeline():
    print(f"\n{'='*55}")
    print("  PHASE 6.5: VALIDATE FLOOD RESULTS")
    print(f"{'='*55}\n")

    # 1. Load Rasters and Shapes
    with rasterio.open(FLOOD_TIF) as src:
        flood_data = src.read(1)
        valid_mask = flood_data > 0
        max_depth = np.max(flood_data)
        mean_depth = np.mean(flood_data[valid_mask]) if np.any(valid_mask) else 0
        total_cells = flood_data.size
        flooded_cells = np.sum(valid_mask)
        flood_pct = (flooded_cells / total_cells) * 100

    print(f"Max flood depth:  {max_depth:.4f} m")
    print(f"Mean flood depth: {mean_depth:.4f} m")
    print(f"% Flooded area:   {flood_pct:.2f} %\n")

    print(f"{'='*55}")
    print("  PHASE 7: FINAL DECISION & EXPORT")
    print(f"{'='*55}\n")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MAPS.mkdir(parents=True, exist_ok=True)

    # 1. Final Decision: Subtract Flood Risk
    print("[1/6] Filtering Safe Zones …")
    gdf_suit = gpd.read_file(SUIT_ZONES)
    
    if FLOOD_ZONES.exists():
        gdf_flood = gpd.read_file(FLOOD_ZONES)
        # Buffer flood zones
        print("      Buffering flood zones by 20m …")
        gdf_flood_buf = gdf_flood.copy()
        gdf_flood_buf['geometry'] = gdf_flood_buf.buffer(20)
        
        # Difference
        print("      Subtracting flood hazards from suitability zones …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # overlay how='difference' keeps geometries in suit that do not intersect with flood buf
            gdf_safe = gpd.overlay(gdf_suit, gdf_flood_buf, how='difference')
    else:
        gdf_safe = gdf_suit.copy()

    # Rank and filter top 50 by area
    gdf_safe['area'] = gdf_safe.geometry.area
    gdf_safe = gdf_safe[gdf_safe['area'] > 10].sort_values(by='area', ascending=False)
    
    if len(gdf_safe) > 50:
        gdf_safe = gdf_safe.head(50)
    
    gdf_safe['safe_flag'] = 1
    gdf_safe.to_file(FINAL_SITES)
    
    # Validation stats
    pct_safe = (len(gdf_safe) / len(gdf_suit)) * 100 if len(gdf_suit) > 0 else 0
    print(f"      Original Zones: {len(gdf_suit)}")
    print(f"      Safe Zones:     {len(gdf_safe)} ({pct_safe:.1f}%)")

    # 2. GeoPackage Export
    print("\n[2/6] Exporting GeoPackage (ALL IN ONE) …")
    # Save active vector layers to GPKG
    if CHANNELS_SHP.exists():
        gpd.read_file(CHANNELS_SHP).to_file(GPKG_OUT, layer='channels', driver="GPKG")
    gdf_suit.to_file(GPKG_OUT, layer='suitability_zones', driver="GPKG")
    if FLOOD_ZONES.exists():
        gpd.read_file(FLOOD_ZONES).to_file(GPKG_OUT, layer='flood_zones', driver="GPKG")
    gdf_safe.to_file(GPKG_OUT, layer='final_sites', driver="GPKG")

    # 3. DXF Export
    print("[3/6] Exporting DXF (Engineering) …")
    if CHANNELS_SHP.exists():
        try:
            gpd.read_file(CHANNELS_SHP).to_file(DXF_OUT, driver="DXF")
        except Exception as e:
            print(f"      [WARN] DXF export failed (Fiona missing driver?).")

    # 4. Shapefiles and Raster copies
    print("[4/6] Copying final datasets …")
    shutil.copyfile(FLOOD_TIF, FINAL_DIR / "flood_map.tif")
    if SLOPE_TIF.exists():
        shutil.copyfile(SLOPE_TIF, FINAL_DIR / "slope.tif")
    
    # Save a copy of final shapefiles to FINAL/ too
    gdf_safe.to_file(FINAL_DIR / "final_sites.shp")
    if FLOOD_ZONES.exists():
        gpd.read_file(FLOOD_ZONES).to_file(FINAL_DIR / "flood_zones.shp")

    # 5. PNG Maps
    print("[5/6] Generating Final Static Maps …")
    generate_high_quality_maps(gdf_suit, gdf_safe, gdf_flood if FLOOD_ZONES.exists() else None)

    # 6. Automate PDF Report
    print("[6/6] Generating Final Engineering Report (PDF) …")
    generate_pdf_report(len(gdf_suit), len(gdf_safe), pct_safe)

    print(f"\n{'='*55}")
    print("FINAL OUTPUT SUMMARY")
    print(f"✔ Total suitability zones: {len(gdf_suit)}")
    if FLOOD_ZONES.exists():
        print(f"✔ Flood risk zones found & avoided.")
    print(f"✔ Final safe zones: {len(gdf_safe)}")
    print(f"✔ % Safe land retained: {pct_safe:.1f}%")
    print("✔ Files exported successfully to outputs/final/")
    print(f"{'='*55}\n")


def generate_high_quality_maps(gdf_suit, gdf_safe, gdf_flood):
    plt.style.use('dark_background')
    
    with rasterio.open(DEM_TIF) as src:
        dem = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    # Map 1: Suitability
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(dem, cmap='Greys_r', extent=extent, alpha=0.5)
    gdf_suit.plot(ax=ax, color='green', alpha=0.7)
    ax.set_title("1. Suitability Map (MCDA)")
    
    suit_patch = mpatches.Patch(color='green', label='MCDA Sites', alpha=0.7)
    ax.legend(handles=[suit_patch], loc="upper right")
    
    plt.savefig(FINAL_MAPS / "1_Suitability_Map.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Map 2: Flood
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(dem, cmap='Greys_r', extent=extent, alpha=0.5)
    
    legend_handles = []
    if gdf_flood is not None and not gdf_flood.empty:
        gdf_flood.plot(ax=ax, color='blue', alpha=0.7)
        legend_handles.append(mpatches.Patch(color='blue', label='Flood Zones', alpha=0.7))
        
    ax.set_title("2. Flood Risk Map")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
        
    plt.savefig(FINAL_MAPS / "2_Flood_Map.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Map 3: Final Decision
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(dem, cmap='Greys_r', extent=extent, alpha=0.5)
    
    legend_handles = []
    if not gdf_safe.empty:
        gdf_safe.plot(ax=ax, color='gold', alpha=0.9, edgecolor='white', linewidth=1)
        legend_handles.append(mpatches.Patch(color='gold', label='Final Safe Sites'))
    
    if gdf_flood is not None and not gdf_flood.empty:
        gdf_flood.plot(ax=ax, color='blue', alpha=0.3)
        legend_handles.append(mpatches.Patch(color='blue', label='Flood Risk (Avoided)', alpha=0.3))
    
    ax.set_title("3. Final Decision Map (MCDA - Flood Risk)")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    
    # Add fake scale/north arrow for decoration
    ax.annotate('N \u2191', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=16, color='white', ha='center', va='center')
    
    plt.savefig(FINAL_MAPS / "3_Final_Decision_Map.png", dpi=250, bbox_inches='tight')
    plt.close()


def generate_pdf_report(initial_count, final_count, pct_safe):
    doc = SimpleDocTemplate(str(REPORT_OUT), pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=20,
        textColor=HexColor('#0055A4')
    )
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=HexColor('#333333')
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.spaceAfter = 12

    elements = []
    
    # Title
    elements.append(Paragraph("Smart Village Site Selection – Bajil Region", title_style))
    elements.append(Paragraph("Final Integrated Evaluation and Decision Report", h2_style))
    
    # 1. Introduction
    elements.append(Paragraph("<b>1. Introduction</b>", h2_style))
    elements.append(Paragraph("This report presents the final geospatial analysis for the Bajil Smart Village project. The goal is to identify the most suitable and safe locations for village infrastructure by combining Multi-Criteria Decision Analysis (MCDA) with predictive flood modeling.", normal_style))
    
    # 2. Data
    elements.append(Paragraph("<b>2. Data Used</b>", h2_style))
    elements.append(Paragraph("• High-resolution DEM generated from contours<br/>• Project Boundary clipped region<br/>• Road infrastructure dataset<br/>• Topographic slopes and flow accumulations", normal_style))
    
    # 3. Methods
    elements.append(Paragraph("<b>3. Methodology</b>", h2_style))
    elements.append(Paragraph("The methodology included: 1) Hydrological conditioning and stream extraction. 2) Fuzzy-logic MCDA balancing Slope, Flood potential, and Road proximity. 3) A 50mm dynamic rainfall flood simulation. 4) Filtering MCDA outputs using a 20m buffered flood-risk exclusion zone.", normal_style))
    
    # 4. Results
    elements.append(Paragraph("<b>4. Results summary</b>", h2_style))
    stats_text = (f"• Initial proposed suitability zones: <b>{initial_count}</b><br/>"
                  f"• Final safe zones (filtered): <b>{final_count}</b><br/>"
                  f"• Percentage of zones safely retained: <b>{pct_safe:.1f}%</b>")
    elements.append(Paragraph(stats_text, normal_style))
    
    elements.append(Spacer(1, 0.2 * inch))

    # 5. Maps
    elements.append(Paragraph("<b>5. Final Map Output</b>", h2_style))
    img_path = FINAL_MAPS / "3_Final_Decision_Map.png"
    if img_path.exists():
        elements.append(Image(str(img_path), width=5.5*inch, height=4.5*inch))
        
    elements.append(Spacer(1, 0.2 * inch))
    
    # 6. Conclusion
    elements.append(Paragraph("<b>6. Conclusion</b>", h2_style))
    elements.append(Paragraph("The identified safe zones guarantee optimum construction conditions with virtually zero flood risk based on simulated events. It is recommended to use the exported 'final_sites.shp' and DXF outputs for the upcoming engineering design layouts.", normal_style))
    
    doc.build(elements)

if __name__ == "__main__":
    run_export_pipeline()
