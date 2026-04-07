# Smart Agri-Village GIS Analysis 🌿🛰️

### [باللغة العربية]
مشروع متطور للتحليل الجيومكاني والتخطيط لمشروع **القرية النموذجية** (بجيل - اليمن). يهدف المشروع إلى استخدام تقنيات الذكاء الاصطناعي ونظم المعلومات الجغرافية (GIS) لدعم الزراعة المستدامة الذكية.

#### الخصائص الحالية:
*   **توليد نموذج الارتفاع الرقمي (DEM):** تحويل خطوط الكنتور (`contor.shp`) إلى راستر عالي الدقة (10 متر) تلقائياً.
*   **المعالجة المسبقة الآلية:** اكتشاف حقول الارتفاع ومعالجة البيانات المفقودة.

---

### [English]
An advanced geospatial analysis and planning project for the **Model Village** (Bajil, Yemen). This project utilizes GIS and AI techniques to support smart sustainable agriculture.

#### Current Features:
*   **Automated DEM Generation:** Converts contour line shapefiles (`contor.shp`) into high-resolution (10m) Digital Elevation Models.
*   **Automatic Preprocessing:** Dynamically detects elevation fields and handles data gaps using interpolation.

---

## 📂 Project Structure
```text
project/
├── data/
│   ├── raw/          # Original GIS data (Contours, Satellite, etc.)
│   └── processed/    # Generated Rasters (DEM, Slopes)
├── src/              # Python source code
│   ├── preprocess.py # Data interpolation & transformation
│   └── main.py       # Main entry point
└── requirements.txt  # Project dependencies
```

## 🚀 Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis:**
   ```bash
   python main.py
   ```

---
**Developed for:** Smart Sustainable Agriculture Projects in Yemen.
