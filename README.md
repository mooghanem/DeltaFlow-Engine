# DeltaFlow-Engine 🌊🛰️

> **A Matrix-Based Digital Elevation and Gravitational Runoff Simulation Engine for Urban Flood Mitigation and Spatial Drainage Optimization in Deltaic Regions**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NSyZo7OWCoHcCcn3an4QB8sm_qhXxEZV?usp=sharing)

---

## 📌 Overview
**DeltaFlow-Engine** is an advanced spatial simulation and geo-referencing engine built in Python. Designed specifically for deltaic terrain regions (with a primary case study in **El-Hamoul, Kafr El-Sheikh, Egypt**), it bridges the gap between matrix-based hydrological modeling and real-world municipal execution by translating flood hazard zones into actionable GPS coordinates and direct Google Maps query links.

## 🛠️ Key Features
* **Real SRTM Topography Integration:** Parses and cleans raw 30m Global Digital Elevation Model (DEM) GeoTIFF data.
* **Empirical Climate Data Processing:** Extracts peak precipitation design parameters safely from NASA POWER climatological datasets.
* **Gravitational Runoff Simulation:** Employs NumPy gradient vector matrices to route surface water downhill following physical mass conservation laws.
* **Statistical Hazard Thresholding:** Automatically isolates critical high-risk pooling zones based on local distribution percentiles.
* **GIS Spatial Mapping & Geo-Referencing:** Instantly translates grid matrix indices into real-world Latitude & Longitude coordinates with direct Google Maps integration.

## 🗂️ Repository Structure
```text
DeltaFlow-Engine/
│
├── DeltaFlow_Engine.ipynb       # Interactive Jupyter Notebook for presentation & demo
├── deltaflow_engine.py          # Standalone Python script / core engine library
├── output_SRTMGL1.tif           # Raw SRTM DEM raster data for El-Hamoul
├── POWER_Climatic_...xlsx       # NASA POWER climatological dataset
└── README.md                    # Project documentation****

---

## 👨‍💻 Author
**Mohamed Khaled Ghanem**  
*Developer, Data Analyst & AI Researcher*  
* 📧 **Email:** [mohammedkhaledfarag@gmail.com](mailto:mohammedkhaledfarag@gmail.com)
* 💼 **LinkedIn:** [linkedin.com/in/mohamedkhaledghanem](https://www.linkedin.com/in/mohamedkhaledghanem/)

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
