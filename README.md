# Smart Agri-Village GIS Analysis 🌿🛰️

### [باللغة العربية]
مشروع متطور للتحليل الجيومكاني والتخطيط لمشروع **القرية النموذجية** (بجيل - اليمن).
يهدف إلى استخدام تقنيات نظم المعلومات الجغرافية (GIS) لدعم الزراعة المستدامة الذكية.

---

### [English]
An advanced geospatial analysis and planning project for the **Model Village** (Bajil, Yemen).
Uses GIS and Python to support smart sustainable agriculture decision-making.

---

## ✅ Current Status: DEM Ready for Hydrology Analysis

```
All checks passed. DEM is scientifically valid. ✓
CRS     : EPSG:32638 (WGS 84 / UTM zone 38N)
Range   : 293 – 364 m | Mean: 312 m | Std: 10.96 m
NoData  : 0.0%
Res     : 10 × 10 m
```

---

## 🔧 Pipeline Functions

| Function | Description |
|----------|-------------|
| `generate_dem_from_contours()` | Contour → 10m DEM via scipy interpolation |
| `inspect_dem()` | Print metadata + Plotly interactive visualization |
| `clip_dem()` | Clip DEM to study boundary (auto CRS fix) |
| `run_validation()` | Full validation suite (see below) |

### Validation Suite (`run_validation`)
| Part | Function | Output |
|------|----------|--------|
| 1 | `validate_crs()` | CRS table + overlap check for all layers |
| 2 | `check_dem_quality()` | Stats, spike detection, nodata %, aspect ratio |
| 3 | `plot_dem()` | matplotlib dark-mode DEM → `dem_check.png` |
| 4 | `check_alignment()` | matplotlib DEM + boundary overlay → `dem_boundary_check.png` |

---

## 📂 Project Structure
```text
project/
├── data/
│   ├── raw/
│   │   ├── contor.shp              # Contour lines (540 features, EPSG:32638)
│   │   └── باب_الناقة.shp          # Study area boundary
│   └── processed/
│       ├── dem.tif                 # Generated DEM 264×69 @ 10m
│       └── dem_clipped.tif         # Clipped DEM
├── outputs/
│   └── figures/
│       ├── dem_*.html              # Plotly interactive (2D + 3D)
│       ├── dem_*.png               # Plotly static PNG
│       ├── dem_check.png           # matplotlib validation plot
│       └── dem_boundary_check.png  # matplotlib alignment overlay
├── src/
│   └── preprocess.py              # All pipeline & validation functions
├── main.py                        # Entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python main.py
```

### Full Pipeline Order:
```
[Step 1] generate_dem_from_contours()  →  dem.tif
[Step 2] inspect_dem()                 →  Plotly HTML + PNG
[Step 3] clip_dem()                    →  dem_clipped.tif + Plotly HTML + PNG
[Step 4] run_validation()              →  CRS check + quality check + matplotlib plots
```

---

## 📦 Dependencies
```
numpy | geopandas | rasterio | scipy
plotly | matplotlib | kaleido==0.2.1 | pyyaml
```

> **Note:** `kaleido==0.2.1` is required for Plotly PNG export (compatible with Plotly 5.x)

---

## 🔜 Upcoming Modules
| File | Content |
|------|---------|
| `src/flow.py` | Slope, flow direction, flow accumulation |
| `src/suitability.py` | Multi-criteria agricultural suitability mapping |
| `src/flood.py` | Flood zone modeling |
| `src/export.py` | PDF report & final map export |

---
**Developed for:** Smart Sustainable Agriculture Projects – Bajil, Yemen.
