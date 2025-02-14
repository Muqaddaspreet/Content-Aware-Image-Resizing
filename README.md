# Content-Aware Image Resizing

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Usage Instructions](#usage-instructions)
- [Results and Comparisons](#results)
  - [Energy Maps](#energy-maps)
  - [Horizontal Resizing Comparisons](#horizontal-resizing-comparisons)
  - [Vertical Resizing Comparisons](#vertical-resizing-comparisons)
- [Future Work](#future-work)
- [Research Paper](#research-paper)
- [References](#references)
- [Contact](#contact)

---

## 👀 Overview

### 📌 Comparing e1, eEntropy, and Gradient Energy Functions

This repository contains Python scripts for **Content-Aware Image Resizing**, comparing three different **energy functions**:  
✅ **e1 Energy Function**  
✅ **eEntropy Energy Function**  
✅ **Gradient Energy Function**  

Each energy function is used to generate an **energy map**, which helps in **image resizing operations** such as:  
- **Carving Horizontally**
- **Enlarging Horizontally**
- **Carving Vertically**
- **Enlarging Vertically**
- **Custom Resizing**

For **gradient-based resizing**, additional **techniques** are explored:  
✔ **Crop and Upscaling**  
✔ **Pixel Updation (Column-wise, Row-wise, Global)**  
✔ **Content-Aware Seam Removal**  

---

## 📂 Project Structure

```bash
.
├── README.md                         # Project documentation
├── scripts/                           # Python scripts for energy functions & resizing
│   ├── e1_energy.py                   # e1 energy function implementation
│   ├── eEntropy_energy.py             # eEntropy energy function implementation
│   ├── gradient_energy.py             # Gradient energy function implementation
│   ├── seam_carving.py                # Seam carving technique
│   ├── crop_upscale.py                # Cropping & upscaling method
│   ├── pixel_update_col.py            # Column-wise pixel update
│   ├── pixel_update_row.py            # Row-wise pixel update
│   ├── pixel_update_global.py         # Global pixel modification
│   ├── content_aware_seam.py          # Content-aware seam removal method
│   └── utils.py                        # Helper functions
├── images/                            # Contains all images used in the project
│   ├── original/                      # Original images before resizing
│   ├── energy_maps/                    # Energy maps for each function
│   ├── carved/                         # Images after seam carving
│   ├── enlarged/                       # Images after enlargement
│   ├── resized/                        # Custom resized images
│   ├── technique_comparison/           # Comparison of different resizing techniques
│   └── results/                         # Final images and outputs
├── requirements.txt                   # Dependencies list
├── Image_Resizing_Techniques.pdf       # Research paper
└── LICENSE                             # License information
```

---

## 🎯 Methodology

### 🔹 Energy Functions
1. **e1 Energy Function**: Measures pixel intensity difference in x and y directions.  
2. **eEntropy Energy Function**: Uses local entropy to assess image complexity.  
3. **Gradient Energy Function**: Uses Sobel filters to detect edges, creating a more precise energy map.  

### 🔹 Image Resizing Techniques
- **Seam Carving** (Content-aware removal/addition of low-energy pixels)  
- **Column-wise and Row-wise Pixel Updation** (Selective modification based on energy values)  
- **Global Pixel Modification** (Adjusts pixels across the entire image)  
- **Cropping & Upscaling** (Simple interpolation-based resizing)  

The goal is to **compare how well each energy function preserves image details** while performing different resizing tasks.

---

## 🖥️ Usage Instructions

### 🔧 **1. Install Dependencies**
Make sure you have Python installed. Then, install the required libraries using:

```bash
pip install -r requirements.txt
```

### 🔧 **2. 📜 2. Run the Scripts**
To compute an energy map and perform seam carving:

```bash
python scripts/seam_carving.py --input images/original_image.jpg --energy e1
```

## 📊 Results & Comparisons

### 🔹 Energy Maps

### 🔹 Horizontal Resizing Comparisons

### 🔹 Vertical Resizing Comparisons

## 🔮 Future Work

📌 **Expand Techniques:** Implement **Wavelet-Based Seam Carving** to improve resizing performance.
📌 **Performance Optimization:** Reduce computational time for entropy-based resizing methods.
📌 **Object-Aware Resizing:** Integrate **object segmentation** to protect key image features while resizing.
📌 **Real-Time Implementation:** Adapt the algorithm for **video resizing** in real-time.

## 📖 Research Paper
For a **detailed explanation** of the algorithms, methodology, and results, read the full research paper:
📄 **[Read the Full Research Paper](Image_Resizing_Techniques.pdf)**

## 📜 References
1. S. Avidan and A. Shamir, “**Seam Carving for Content-Aware Image Resizing**”, ACM Trans. Graph., 2007.
2. J.-W. Han, K.-S. Choi, T.-S. Wang, S.-H. Cheon, and S.-J. Ko, “**Wavelet-Based Seam Carving for Content-Aware Image Resizing**”, IEEE, 2010.

## 👤 Contact
👨‍💻 **Muqaddaspreet Singh Bhatia**
📫 Email: *(muqaddaspreetsb@gmail.com)*
🌐 GitHub: [Muqaddaspreet](https://github.com/Muqaddaspreet)

🚀 If you found this project useful, **consider starring ⭐ the repo!**


