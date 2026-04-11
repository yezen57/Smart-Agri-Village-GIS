"""
main.py
=======
Entry point for the Bajil Smart Agricultural Village – GIS pipeline.

Usage
-----
    python main.py
"""

from src.preprocess import (
    generate_dem_from_contours,
    inspect_dem,
    clip_dem,
    validate_crs,
    check_dem_quality,
    prepare_roads_layer,
    clip_vector_boundary_to_raster,
    run_validation,
    BOUNDARY_PATH
)
from src.hydrology import run_hydrology_pipeline
from src.watershed import run_flow_analysis, run_sensitivity_analysis
from src.suitability import run_suitability_analysis
from src.flood import run_flood_simulation
from src.export import run_export_pipeline
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent



def main():
    """
    Executes the full Pipeline:
    # 1. DEM Gen -> 2. Inspection -> 3. Clipping -> 
    # 4. Roads -> 5. Vector Mapping -> 6. Validation -> 
    # 7. Hydrology (Slope/Aspect) -> 8. Watershed (Channels) ->
    # 9. Sensitivity -> 10. Suitability -> 11. Flood Simulation -> 12. Export & Decisions
    """
    print("Starting GIS preprocessing pipeline …\n")

    # ── Step 1: Generate DEM from contour lines ──────────────────────────
    dem_path = generate_dem_from_contours()
    print(f"\n✓ DEM ready at: {dem_path}\n")

    # ── Step 2: Inspect the generated DEM (Plotly) ───────────────────────
    inspect_dem(dem_path, title="Generated DEM")

    # ── Step 3: Clip DEM to Boundary ──────────────────────────────────────
    clipped_path = clip_dem(dem_path=dem_path)

    # ── Step 4: Prepare Roads Layer ───────────────────────────────────────
    roads_in  = PROJECT_ROOT / "data" / "الطرق" / "roads.shp"
    roads_out = PROJECT_ROOT / "data" / "processed" / "roads_clipped.shp"
    prepare_roads_layer(roads_in, boundary_path=BOUNDARY_PATH, output_path=roads_out)

    # ── Step 5: GIS & Vector Mapping ──────────────────────────────────────
    bound_clip_path = clip_vector_boundary_to_raster()
    print(f"\n✓ Clipped Boundary (Vector) ready at: {bound_clip_path}\n")

    # ── Step 6: Full Validation Suite ────────────────────────────────────
    run_validation()

    # ── Step 6: Hydrological Processing (Fill Sinks, Slope, Aspect) ───────
    run_hydrology_pipeline(dem_path=clipped_path)

    # ── Step 7: Flow Analysis (Direction, Accumulation, Streams) ──────────
    run_flow_analysis(threshold_pct=0.5)

    # ── Step 8: Sensitivity Analysis (Comparison of 3 Thresholds) ─────────
    run_sensitivity_analysis()

    # ── Step 10: Suitability Analysis (Final Site Selection) ──────────────
    run_suitability_analysis()

    # ── Step 11: Flood Simulation (Rainfall 50mm) ────────────────────────
    run_flood_simulation(rainfall_mm=50)

    # ── Step 12: Final Decision & Export Pipeline ────────────────────────
    run_export_pipeline()


if __name__ == "__main__":
    main()
