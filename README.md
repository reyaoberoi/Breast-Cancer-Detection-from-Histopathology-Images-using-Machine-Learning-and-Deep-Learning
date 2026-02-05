# 🧬 Breast Cancer Detection from Histopathology Images

### Machine Learning & Deep Learning Approaches

## 📌 Overview

Breast cancer remains one of the leading causes of cancer-related mortality among women worldwide. Histopathological image analysis is the gold standard for diagnosis, but manual examination is time-consuming and subject to inter-observer variability.

This repository presents a **comprehensive comparative study of classical Machine Learning (ML) and Deep Learning (DL) approaches** for breast cancer classification using histopathology images from the **BreaKHis dataset**. The work emphasizes **accuracy, statistical rigor, computational efficiency, and clinical relevance**.

---

## 📂 Projects Included

### 1. Classical Machine Learning Analysis

* Feature-based ML classification using:

  * HSV color histograms
  * Haralick texture features (GLCM)
* Models evaluated:

  * XGBoost
  * Support Vector Machine (SVM)
  * K-Nearest Neighbors (KNN)
  * Logistic Regression
  * Decision Tree
  * Linear Discriminant Analysis (LDA)
* Statistical validation:

  * 5-fold cross-validation
  * One-way ANOVA
  * Bonferroni-corrected pairwise t-tests

📈 **Best ML Performance:**

* **XGBoost – 94.9% accuracy, 96.26% F1-score**

---

### 2. Deep Learning & Transfer Learning Analysis

* End-to-end image classification using CNNs and pretrained architectures:

  * Custom CNN
  * ResNet50
  * DenseNet121
  * EfficientNetV2-B0
  * MobileNetV1
  * InceptionV3
* Transfer learning with ImageNet weights
* Evaluation across multiple magnification levels (40×–400×)

📈 **Best DL Performance:**

* **EfficientNetV2-B0 – 91.41% accuracy with highest computational efficiency**
* **ResNet50 – Highest recall (97.23%), critical for medical diagnosis**

---

## 🧠 Methodology Summary

1. **Dataset**

   * BreaKHis histopathology dataset
   * Benign vs Malignant classification
   * Multiple magnification levels (40×, 100×, 200×, 400×)

2. **Preprocessing**

   * Image resizing and normalization
   * Data augmentation (rotation, flips, contrast variation)

3. **Feature Engineering (ML)**

   * HSV color histograms
   * GLCM-based Haralick texture descriptors

4. **Model Training**

   * Consistent hyperparameters
   * GPU-accelerated training (Google Colab)

5. **Evaluation**

   * Accuracy, Precision, Recall, F1-score
   * Statistical significance testing

---

## 📊 Key Results

| Approach         | Best Model        | Accuracy   |
| ---------------- | ----------------- | ---------- |
| Machine Learning | XGBoost           | **94.9%**  |
| Deep Learning    | EfficientNetV2-B0 | **91.41%** |

 * Transfer learning models consistently outperformed custom CNNs

 * Ensemble ML models demonstrated strong robustness and interpretability

 * EfficientNetV2-B0 offered the best trade-off between accuracy and speed

---

## 🛠️ Tech Stack

* Python
* NumPy, OpenCV
* Scikit-learn
* XGBoost
* TensorFlow / Keras
* Matplotlib, Seaborn

---


## 🚀 Future Work

* Explainable AI (Grad-CAM, SHAP) for clinical interpretability
* Multi-magnification and whole-slide image analysis
* Ensemble DL architectures
* Model compression for real-time deployment

---

## 👩‍🔬 Authors

* **Reya Oberoi**
* Maanasvee Khetan
* Sanya Malik
