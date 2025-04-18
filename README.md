# ML-Playground

## 🚀 Machine Learning Algorithms from Scratch in Python

Welcome to **ML-Playground**, a collection of core machine learning algorithms implemented from scratch using Python.  
Perfect for beginners, students, and anyone who wants to understand the inner workings of ML models.

---

## 🌟 Featured Algorithms

### 1. 🔹 Linear Regression

**Implemented using Gradient Descent**

**Key Features:**
- 💡 Flexible execution using `ArgumentParser`
- 🛠️ Preprocessing options:
  - `0`: No preprocessing  
  - `1`: Min/Max scaling  
  - `2`: Standardization
- 🧠 Training choices:
  - `0`: Verify a perfect 45-degree linear line  
  - `1`: Train with all features  
  - `2`: Train with the best feature  
  - `3`: Solve using Normal Equations  
  - `4`: Solve using Scikit-Learn
- ⚙️ Adjustable parameters:
  - `--step_size`: Learning rate  
  - `--precision`: Requested precision  
  - `--max_iter`: Maximum number of iterations (epochs)  
  - `--dataset`: Path to dataset file

---

### 2. 🔹 K-Nearest Neighbors (KNN) Classifier

**From-scratch implementation of a classic classification algorithm**

**Key Features:**
- 📏 Distance metrics supported:
  - ✅ Euclidean distance (`euclidean`)
  - ✅ Cosine similarity (`cosine`) 
  - ✅ Manhattan distance (`manhattan`)
- 🔢 Flexible neighbor selection with adjustable `k` value
- 🌈 Visualizations of decision boundaries
- 🌼 High accuracy (95%+) on the Iris dataset

**Example Usage:**
```bash
python KNN/main.py --k 5 --distance euclidean
