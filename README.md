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


---

### 3. 🔹 Linear & Logistic Regression 

**Notebook:** `LinearRegression&LogisticRegression.ipynb`  
An interactive exploration of regression using multiple optimization strategies.

**What’s inside:**
- 📊 Data generation and splitting
- 📉 **Linear Regression** with:
  - Batch Gradient Descent  
  - Stochastic Gradient Descent  
  - Mini-batch Gradient Descent
- 🔐 **Logistic Regression** with:
  - Batch Gradient Descent  
  - Stochastic Gradient Descent  
  - Mini-batch Gradient Descent
 
  ---

### 4. 🔹 Voice Gender Classification

A machine learning system that classifies voice recordings as male or female using audio features and ensemble learning techniques.

## Project Overview

This project demonstrates:
- Audio feature extraction (MFCCs, spectral features)
- Custom Gaussian Naive Bayes implementation
- Model comparison with scikit-learn classifiers
- Ensemble methods with Bagging
- Real-time voice prediction capability

## Key Features

- **Audio Processing**:
  - Noise reduction and silence removal
  - MFCC, spectral centroid, rolloff, and ZCR feature extraction
- **Machine Learning Models**:
  - Custom Gaussian Naive Bayes (80% accuracy)
  - Scikit-learn's GaussianNB (80% accuracy)
  - Logistic Regression (86.7% accuracy)
- **Ensemble Learning**:
  - Bagging classifiers with majority voting
  - Achieves 83.3% accuracy

## Results Summary

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Custom NB              | 0.8000   | 0.8000    | 0.8000 | 0.8000   |
| Logistic Regression    | 0.8667   | 0.9231    | 0.8000 | 0.8571   |
| Bagging (Logistic)     | 0.8667   | 1.0000    | 0.7333 | 0.8462   |


 
---

