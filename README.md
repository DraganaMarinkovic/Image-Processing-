# OS 2024 – Project: Image Processing with CUDA  

## Overview  
This project implements three basic image processing algorithms using **CUDA / PyCUDA**:  
1. **Grayscale conversion**  
2. **Gaussian blur**  
3. **Brightness adjustment**  

The goal is to parallelize pixel operations efficiently across the GPU.  

---

## Algorithms  

### 1. Grayscale (6 pts)  
- Convert an RGB image to grayscale using weighted sum:  
  `0.299 * R + 0.587 * G + 0.114 * B`  
- Produces a single-channel image with values in `[0, 255]`.  
- Optimized for coalesced memory access across different image sizes.  

### 2. Gaussian Blur (8 pts)  
- Apply a **Gaussian filter** to each channel (R, G, B) separately.  
- Each pixel is replaced by a weighted sum of neighbors using a Gaussian kernel.  
- Kernel values precomputed on the CPU (Python).  
- Parallelization across channels using block’s **z-dimension**.  
- Requirements:  
  - Kernel size parameterized (e.g., 3×3, 5×5, 7×7).  
  - Must work on images larger than a single block.  
  - Shared/constant memory used where applicable.  

### 3. Brightness Adjustment (6 pts)  
- Two kernels:  
  1. Compute mean pixel intensity.  
  2. Scale pixel values relative to mean by a user-provided factor.  
- Ensures pixel values stay in range `[0, 255]`.  
- Reduction kernel for computing sum should minimize warp divergence and memory bank conflicts.  

---

## Implementation Principles  
- Each thread handles **one pixel**.  
- Threads arranged across **x, y** block dimensions.  
- Results written to an output image array.  
- Images loaded as **NumPy arrays**, then transferred to CUDA memory.  
- Functions implemented separately for each algorithm.  

---

## Gaussian Blur Details  
1. **Kernel Creation**  
   - Gaussian matrix `N×N` (e.g., 3×3, 5×5).  
   - Values based on Gaussian distribution:  
     ```
     G(x,y) = (1 / (2πσ²)) * e^(-(x² + y²) / (2σ²))
     ```  
2. **Normalization**  
   - All kernel values divided by their total sum (preserves brightness).  
3. **Convolution**  
   - For each pixel: align kernel, multiply and sum overlapping values.  
4. **Edge Handling** (choose one and document):  
   - Zero padding  
   - Mirror padding  
   - Edge repeat padding  

---
