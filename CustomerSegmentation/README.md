# 🧮 Customer Segmentation using KMeans

This project applies **unsupervised machine learning** to segment mall customers based on their behavior and demographics. The goal is to identify distinct customer groups for targeted marketing.

---

## 📊 Dataset

- **Source**: `Mall_Customers.csv`
- **Features Used**:
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`
  - `Gender` (one-hot encoded)

---

## 🧠 ML Technique

- **KMeans Clustering**:
  - Used to group customers into clusters based on similar characteristics.
  - Number of clusters determined using the **Elbow Method**.

---

## 🛠️ Workflow

1. **Preprocessing**
   - Dropped `CustomerID`
   - One-hot encoded `Gender`
   - Scaled features using `StandardScaler`

2. **Clustering**
   - Applied KMeans with `k=5` (based on elbow plot)
   - Predicted cluster labels

3. **Post-processing**
   - Inverse scaled data for interpretability
   - Assigned cluster labels to unscaled data

---

## 📈 Visualizations

- **Elbow Plot** to identify optimal number of clusters
- **2D Scatter Plot**: `Age` vs `Spending Score` by cluster
- **3D Scatter Plot**: `Age`, `Annual Income`, `Spending Score` colored by cluster

---

## 🧰 Tools & Libraries

- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  

---

## 📁 Folder Structure

