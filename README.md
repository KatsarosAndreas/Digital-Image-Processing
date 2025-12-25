# Image Processing & Feature Extraction

Practical implementations of classical and modern image analysis techniques—foundational skills for computer vision research, medical imaging, and intelligent surveillance systems.

---
# Image Processing – Laboratory Exercises (Part A)

This repository contains the complete implementation, experimentation, and analysis for **Laboratory Exercise 1** of the *Image Processing* course.  
The work focuses on fundamental and advanced image processing techniques, combining theoretical concepts with practical MATLAB implementations.

The repository is structured to clearly separate each exercise, provide well-documented code, and demonstrate reproducible results suitable for academic evaluation and research-oriented portfolios.

---

## Overview

The exercises cover the following major topics:

- Frequency-domain image processing using the Discrete Fourier Transform (DFT)
- Image compression using the Discrete Cosine Transform (DCT)
- Noise modeling and denoising techniques
- Histogram analysis and contrast enhancement
- Comparative evaluation using quantitative metrics (MSE, PSNR)

All implementations are written in **MATLAB**, using `.mlx` live scripts for clarity, visualization, and step-by-step explanation.

---

## Repository Structure

```text
.
├── ex_1.mlx            # Frequency-domain preprocessing and DFT centering
├── ex_2.mlx            # 2D-DFT implementation and spectrum visualization
├── ex_3.mlx            # Low-pass filtering in the frequency domain
├── ex_4.mlx            # Inverse DFT and image reconstruction
├── ex_5_part_a.mlx     # DCT-based image compression (zonal method)
├── ex_5_part_b.mlx     # DCT-based image compression (threshold method)
├── ex_6.mlx            # Noise filtering and histogram equalization
├── KATSAROS_ANDREAS_1084522_EA1_REPORT.pdf
└── README.md
```
# Image Processing – Laboratory Exercises (Part B)

This repository contains the implementation for **Laboratory Exercise – Part B** of the *Image Processing* course.  
The focus of this part is on **feature extraction** and **pattern classification**, applying classical computer vision techniques to real image data.

The code is intentionally kept simple, clear, and well-structured, emphasizing correctness and reproducibility rather than over-engineering.

---

## Overview

Part B concentrates on object representation and classification using:

- Histogram of Oriented Gradients (HOG)
- Support Vector Machines (SVM)
- Digit classification using the MNIST dataset

The implementation demonstrates a complete pipeline:
data loading → feature extraction → model training → evaluation.

---

## Repository Structure

```text
.
├── hog_svm_mnist_classifier.m   # Main script: HOG feature extraction + SVM training
├── mnist_data_loader.m          # Utility for loading MNIST dataset
├── Image_Processing_PART_A.ipynb
├── KATSAROS_ANDREAS_EA2.pdf     # Technical report (Part B)
└── README.md
```

## Projects
## Institution

**University of Patras**  
**Department of Electrical and Computer Engineering**  
**Student ID:** 1084522  
**Course:** Image Processing

---

*Demonstrating practical computer vision skills for research positions in autonomous systems, medical imaging, and intelligent analytics.*
