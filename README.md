## 🔍 About This Fork

This project extends the original **Real-Time 3D Dashcam Perception System** 
by [tubakhxn](https://github.com/tubakhxn/Real-Time-3D-Dashcam-Perception-System), 
which combines YOLOv8 vehicle detection with monocular depth estimation to 
build a real-time 3D Bird's-Eye-View and collision risk overlay.

**What I added:**
- 📊 A detection logging system that captures per-frame vehicle distance, 
  risk classification, and tracking data to CSV
- 🐍 A Pandas/Seaborn analytics script (`analyze_results.py`) generating 
  risk distribution, distance-over-time, and vehicle-count summaries
- 📈 A Power BI dashboard built on the exported detection log, surfacing 
  risk events and distance trends for analysis

Full credit to the original author for the core CV/ML pipeline — this fork 
focuses on turning the model's raw output into structured, analyzable insight.