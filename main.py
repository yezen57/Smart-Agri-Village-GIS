"""
main.py
=======
Entry point for the Bajil Smart Agricultural Village – GIS pipeline.

Usage
-----
    python main.py
"""

from src.preprocess import generate_dem_from_contours


def main():
    print("Starting GIS preprocessing pipeline …\n")

    # Step 1 – Generate DEM from contour lines
    dem_path = generate_dem_from_contours()
    print(f"\n✓ DEM ready at: {dem_path}\n")


if __name__ == "__main__":
    main()
