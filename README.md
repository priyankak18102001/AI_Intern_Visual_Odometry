# AI_Intern_Visual_Odometry
Debugged and refactored a visual odometry pipeline to accurately estimate camera motion from image sequences using OpenCV.
# Visual Odometry – Debug & Refactor Task

This repository contains a refactored and debugged **Visual Odometry pipeline** that estimates camera motion from image sequences using feature tracking.

## 📌 Overview
The goal of this task was to:
- Fix existing bugs in the visual odometry pipeline
- Refactor the code into clean, modular functions
- Add basic error handling
- Ensure reasonable trajectory accuracy and motion correlation

The pipeline simulates camera motion using a single high-resolution image and estimates relative motion between frames using ORB feature matching.

---

## 🛠️ Tech Stack
- Python
- OpenCV
- NumPy

---

## 📂 Project Structure

---

## ▶️ How to Run

### 1. Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows


pip install -r requirements.txt

python vo_pipeline.py
### Output
Trajectory Error: 0.061
Correlation r: 0.990


