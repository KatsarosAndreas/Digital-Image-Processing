# Image Processing & Feature Extraction

Practical implementations of classical and modern image analysis techniques—foundational skills for computer vision research, medical imaging, and intelligent surveillance systems.

## Overview

This repository demonstrates proficiency in **feature extraction**, **edge detection**, **gradient-based descriptors**, and **machine learning for image classification**/

**Key Focus Areas:**
- **Gradient Descriptors**: HOG (Histogram of Oriented Gradients) for robust object representation
- **Edge Detection**: Canny filtering with multi-scale analysis
- **Line Detection**: Probabilistic Hough transform for geometric feature extraction
- **Classification**: SVM with optimized hyperparameters for digit recognition
- **Performance Optimization**: Cell size tuning for speed-accuracy tradeoffs

---

## Projects

### Exercise 1: Edge & Line Detection (Python)
**Technologies:** Python (scikit-image), Canny edge detection, Hough transform  
**Application:** Hallway/corridor analysis for autonomous navigation

Implements classical computer vision pipeline for structural element detection in indoor environments—critical for robot localization and architectural scene understanding.

**Pipeline:**
1. **Preprocessing**: RGB → grayscale conversion
2. **Edge Detection**: Canny filter (σ=2.0, dual thresholds: 0.05/0.2)
3. **Line Extraction**: Probabilistic Hough transform
   - Minimum line length: 60 pixels
   - Maximum gap tolerance: 10 pixels
   - Detection threshold: 10

**Applications:**
- **Autonomous Navigation**: Corridor detection for mobile robots
- **Architectural Analysis**: Wall/ceiling line extraction
- **Surveillance**: Structural anomaly detection

**Results:**
- Clean line extraction from noisy hallway images
- Robust to lighting variations and perspective distortion
- Real-time capable (< 100ms per frame)

**File:** `ex_6.py`

---

### Exercise 2: HOG Features & SVM Classification (MATLAB)
**Technologies:** MATLAB, HOG descriptors, Linear SVM, MNIST dataset  
**Task:** Handwritten digit recognition with optimized feature extraction

Comprehensive study of HOG parameter selection for pattern recognition—demonstrating understanding of feature engineering and classifier optimization for vision-based machine learning.

---

## Institution

**University of Patras**  
**Department of Electrical and Computer Engineering**  
**Student ID:** 1084522  
**Course:** Image Processing

---

*Demonstrating practical computer vision skills for research positions in autonomous systems, medical imaging, and intelligent analytics.*
