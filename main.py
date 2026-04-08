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
    clip_vector_boundary_to_raster,
    run_validation,
)
from src.hydrology import run_hydrology_pipeline
from src.watershed import run_flow_analysis



def main():
    """
    Executes the full Pipeline:
    1. DEM Gen -> 2. Inspection -> 3. Clipping -> 
    4. Vector Mapping -> 5. Validation -> 
    6. Hydrology (Slope/Aspect) -> 7. Watershed (Channels)
    """
    print("Starting GIS preprocessing pipeline …\n")

    # ── Step 1: Generate DEM from contour lines ──────────────────────────
    dem_path = generate_dem_from_contours()
    print(f"\n✓ DEM ready at: {dem_path}\n")

    # ── Step 2: Inspect the generated DEM (Plotly) ───────────────────────
    inspect_dem(dem_path, title="Generated DEM")

    # ── Step 3: Clip DEM to study boundary ───────────────────────────────
    clipped_path = clip_dem()
    print(f"\n✓ Clipped DEM ready at: {clipped_path}\n")

    # ── Step 4: Clip Vector Boundary to DEM Extent ──────────────────────
    bound_clip_path = clip_vector_boundary_to_raster()
    print(f"\n✓ Clipped Boundary (Vector) ready at: {bound_clip_path}\n")

    # ── Step 5: Full Validation Suite ────────────────────────────────────
    run_validation()

    # ── Step 6: Hydrological Processing (Fill Sinks, Slope, Aspect) ───────
    run_hydrology_pipeline(dem_path=clipped_path)

    # ── Step 7: Flow Analysis (Direction, Accumulation, Streams) ──────────
    run_flow_analysis(threshold_pct=0.5)


if __name__ == "__main__":
    main()
