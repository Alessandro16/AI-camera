# 🚶‍♂️ Real-Time Pedestrian Detection System

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![Tests](https://img.shields.io/badge/Tests-Passed-success)

A high-performance Computer Vision pipeline for real-time pedestrian detection. This project leverages the **YOLOv8** architecture to identify people in static images and video streams, featuring automated hardware optimization for **NVIDIA CUDA**.

---

## 📸 Proof of Concept
To demonstrate the model's accuracy, here is a comparison between a raw input frame and the processed output with bounding boxes and tracking ID.

| Original Input | Detection Result |
| :---: | :---: |
| ![Original](assets/test_image.jpg) | ![Detected](test_results/result_output.jpg) |

---

## 🌟 Key Features
* **Hardware Acceleration**: Automatic detection of **CUDA GPU** with seamless fallback to CPU.
* **Real-Time Performance**: Optimized inference logic suitable for live camera streams.
* **Automated Testing Suite**: Full validation of hardware, media integrity, and model output via **Pytest**.
* **Modular Design**: Decoupled detection logic for easy integration into larger AI pipelines.

---

## 🛠️ Project Structure
```text
.
├── assets/              # Test images and sample videos
├── src/                 # Core detection and logic source code
├── tests/               # Automated test suite (Pytest)
├── models/              # YOLOv8 pre-trained weights (.pt)
├── test_results/        # Output gallery for documentation
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
