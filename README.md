# 📊 Cluster Validation on Time-Series Sensor Data

This project performs unsupervised clustering on sensor-derived features to determine the optimal number of clusters using internal validation metrics. It focuses on K-means and hierarchical clustering, applying time-series feature extraction to generate meaningful representations of input signals.

---

## 📌 Project Overview

- Uses statistical and frequency-based features extracted from sensor signals.
- Applies clustering algorithms across different values of `k` to evaluate grouping quality.
- Analyzes and compares results using internal validation metrics such as:
  - **Silhouette Score**
  - **Davies-Bouldin Index**

---

## 🚀 Features

- Time-series feature extraction:
  - FFT (Fast Fourier Transform) coefficients
  - Minimum, maximum, mean, and standard deviation
  - First and second-order differences

- Clustering:
  - K-means clustering (multiple values of `k`)
  - Agglomerative Hierarchical Clustering

- Validation:
  - Silhouette plots
  - Optimal `k` detection
  - Cluster evaluation metrics and visualizations

---

## 🗂️ File Descriptions

| File                                        | Description                                                                 |
|---------------------------------------------|-----------------------------------------------------------------------------|
| `main.py`                                   | Main script: loads data, extracts features, applies clustering, evaluates results |
| `CSE 572_Project 3 Cluster Validation_Overview Document.pdf` | Detailed project description with methodology and results                  |

---

## 🛠️ How to Run

### **Step 1: Install Dependencies**

Run the following to install the required Python libraries:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

---

### **Step 2: Execute the Clustering Script**

Run the `main.py` file to begin the process:

```bash
python main.py
```

This script will:
- Load the dataset
- Extract time-series features
- Apply clustering over a range of `k` values
- Output performance metrics for each value of `k`
- Display visualizations such as silhouette scores and cluster assignments

---

## 📊 Sample Output

```
Evaluating K-means clustering from K=2 to K=10
Silhouette Score for k=2: 0.55
Silhouette Score for k=3: 0.72 ✅
Silhouette Score for k=4: 0.60
...
Optimal number of clusters based on silhouette: 3
Davies-Bouldin Index for k=3: 0.51
```

---

## ⚠️ Dataset Notice

The dataset used for this analysis contains sensitive data and is not shared in this repository.  
To run the project, you should provide a similar multivariate time-series dataset with consistent timestamps.

---

## 📄 License

This project is intended for educational use. Please cite appropriately if reused or extended.
