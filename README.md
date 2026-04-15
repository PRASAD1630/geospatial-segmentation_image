# geospatial-segmentation_image
# 🌍 Geospatial Feature Extraction using Multi-Model Deep Learning

🚀 An advanced **end-to-end geospatial AI pipeline** designed to extract meaningful spatial features such as **buildings, roads, road centerlines, and water bodies** from ultra-high-resolution satellite imagery and convert them into **GIS-ready vector formats (.gpkg)**.

---

## 🎯 Objective

This project focuses on automating geospatial feature extraction for real-world applications:

* 🏙️ Smart City Planning
* 🛣️ Road Network Analysis
* 🌊 Water Resource Monitoring
* 🌍 GIS-based Spatial Intelligence Systems

---

## 🧠 System Architecture

The pipeline is designed to efficiently process **very large GeoTIFF images (10k × 20k+)** using a scalable and memory-efficient approach.

### 🔄 Workflow

```
GeoTIFF Input
      ↓
Tiling Engine
      ↓
Model Router
      ↓
Deep Learning Models
      ↓
Post-processing
      ↓
GIS Outputs (.tif + .gpkg)
```

---

## 🛰️ Core Components

### 1️⃣ Input Layer

* Accepts ultra-high-resolution **GeoTIFF satellite images**

---

### 2️⃣ Tiling Engine (Memory Optimization)

* Splits large images into smaller overlapping patches
* Enables **GPU-safe inference**
* Prevents memory overflow
* Maintains spatial consistency

---

### 3️⃣ Model Routing System

A dynamic multi-model architecture that selects models based on the task:

| Feature              | Model          | Task                  |
| -------------------- | -------------- | --------------------- |
| 🏢 Buildings         | UNet / CNN     | Building segmentation |
| 🛣️ Roads            | DINOv2-based   | Road segmentation     |
| 🌊 Water Bodies      | DeepLabV3+     | Water segmentation    |
| 🌊 Water Boundaries  | Refinement CNN | Edge extraction       |
| 🛣️ Road Centerlines | Skeleton CNN   | Graph extraction      |

---

### 4️⃣ Post-Processing Layer

Improves prediction quality using:

* Morphological filtering (noise removal)
* Skeletonization (centerline extraction)
* Contour detection
* Polygonization (vector conversion)

---

### 5️⃣ GIS Output Layer

📌 Outputs generated in:

**Raster:**

* `.tif` segmentation masks

**Vector:**

* `.gpkg` files (QGIS / ArcGIS compatible)

---

## ⚡ Key Features

* 🧠 Multi-model deep learning pipeline
* 🛰️ Supports ultra-large satellite images
* 🔲 Sliding window tiled inference
* 🔁 Intelligent model routing
* 🧭 Dual road representation (mask + centerline)
* 🌊 Water boundary refinement
* 🗺️ Raster → Vector GIS conversion
* ⚡ Memory-efficient processing

---

## 📊 Model Performance

| Feature              | IoU Score |
| -------------------- | --------- |
| 🏢 Buildings         | ~81%      |
| 🛣️ Roads            | ~79%      |
| 🌊 Water Bodies      | ~82%      |
| 🌊 Water Lines       | ~82%      |
| 🛣️ Road Centerlines | ~64%      |

---

## 📂 Project Structure

```
Project/
│
├── data/
│   ├── input_images/
│   ├── output/
│
├── src/
│   ├── model_router.py
│   ├── inference_pipeline.py
│   ├── building_model.py
│   ├── road_model.py
│   ├── water_model.py
│   ├── water_line_model.py
│   ├── road_centerline_model.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/PRASAD1630/geospatial-segmentation_image.git
cd geospatial-segmentation_image

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python main.py
```

### 🔄 Execution Steps:

1. Select pipeline mode
2. Load GeoTIFF input
3. Run tiled inference
4. Generate segmentation outputs
5. Export GIS files

---

## 📥 Input

Place GeoTIFF images in:

```
data/input_images/
```

Example:

```
data/input_images/sample.tif
```

---

## 📤 Output

Generated outputs include:

* building_output.tif
* road_output.tif
* water_output.tif
* water_line_output.tif
* road_centerline_output.tif
* GIS vector files (.gpkg)

---

## 🚧 Challenges Addressed

* Handling ultra-high-resolution imagery
* Multi-model coordination
* Memory-efficient inference
* Accurate road centerline extraction
* Raster-to-vector GIS conversion

---

## 🚀 Future Enhancements

* 🌐 Web-based GIS dashboard (Streamlit / React)
* ☁️ Cloud deployment (Docker + FastAPI)
* 📡 Real-time satellite inference API
* 🧭 Road network graph extraction
* ⚡ Batch processing optimization

## 🧑‍💻 Author

**Prasad**
AI & ML Student
Focus: Computer Vision | Geospatial AI | Deep Learning Systems

---

## 🌟 Final Impact

This project demonstrates a **production-ready geospatial intelligence system** that transforms raw satellite imagery into structured GIS data, enabling scalable solutions for urban planning, environmental monitoring, and smart infrastructure development.

---
