# Comparative Analysis of Serial and Parallel Image Processing (CPU & GPU with CUDA + PyTorch)

## Project Description

This project presents a **performance-focused comparative analysis** of five widely-used image processing algorithms implemented on:

- **Serial CPU** (NumPy/SciPy)
- **Parallel CPU** (PyTorch)
- **GPU using CUDA** (via CuPy)

The goal is to analyze the **execution time, speedup, and efficiency** of each implementation across multiple image sizes using Google Colaboratory (Tesla T4 GPU, 2 vCPUs). These algorithms are critical in **medical imaging** such as MRI, CT scan enhancement, and X-ray noise suppression.

---

## Algorithms Implemented

| Algorithm            | Purpose                                  | Parallelized On       |
|---------------------|------------------------------------------|------------------------|
| Gaussian Blur        | Noise reduction using convolution        | CPU + GPU             |
| Sobel Edge Detection | Gradient-based edge detection            | CPU + GPU             |
| Prewitt Filter       | Simplified edge detection (gradient)     | CPU + GPU             |
| Roberts Filter       | Diagonal edge detection (tiny kernels)   | CPU + GPU             |
| Otsu's Thresholding  | Optimal binary segmentation (threshold)  | CPU + GPU             |

---

## Technologies Used

- Python 3.x
- NumPy / SciPy
- PyTorch (for parallel CPU)
- CuPy + Raw CUDA Kernels (for GPU)
- OpenCV (image handling)
- Google Colab (hardware backend)

---




## Performance Metrics

| Metric                     | Description |
|---------------------------|-------------|
| **Execution Time (s)**     | Time taken for each method |
| **Speedup**                | Ratio of serial to parallel time |
| **Efficiency**             | Speedup ÷ number of processing units |


---


## Visual Output

Each algorithm produces comparative output across all backends (Serial CPU, PyTorch CPU, CUDA GPU):

```
+----------------+----------------+----------------+
|  Serial CPU    |  PyTorch CPU   |  CuPy GPU      |
|   (Output)     |   (Output)     |   (Output)     |
+----------------+----------------+----------------+
```

Graphs included:
- Execution Time vs Image Size
- Speedup vs Image Size
- Efficiency vs Image Size

---

## How to Run
1. Open [Google Colab](https://colab.research.google.com)
2. Click on File > Open Notebook
3. In the URL tab, paste your repo notebook link. Example:

   
```bash
   https://github.com/RAMYA-M-08/image_processing_parallel_computing/blob/main/GAUSSIAN_BLUR.ipynb
```


Then replace `github.com` with `colab.research.google.com/github/` like this:


```bash
   https://colab.research.google.com/github/RAMYA-M-08/image_processing_parallel_computing/blob/main/GAUSSIAN_BLUR.ipynb
```

4. Set up the GPU:

- Go to Runtime > Change runtime type

- Set Hardware accelerator to T4 GPU 

- Click Save

5. Run all cells:
   

     Click Runtime > Run all

---

##  Experimental Setup

-  **Google Colab (Free GPU T4)**
-  **2 CPU Cores (Xeon)**
-  **CUDA GPU: Tesla T4 (2560 cores, 40 SMs)**
-  **Images Tested**: 512×512 to 4000×4000
-  **Kernel sizes**: 7×7 to 31×31 (for Gaussian)

---

## Output Highlights

- GPU (CUDA with CuPy) **outperformed** all other methods across all algorithms.

- Speedup for Gaussian Blur peaked at **198x**, while Otsu’s Thresholding GPU variant saw up to **12x** speedup.

- PyTorch CPU gave moderate gains (~2x–4.5x), but struggled with scalability compared to GPU.

- Efficiency showed that GPU consistently utilized its cores better (~4.9 per SM) than CPU (2.2 per core).

---

## Team Members

- [Ramya M ](https://github.com/RAMYA-M-08)
- [Elakiya R](https://github.com/Elakiya-R31)

---

