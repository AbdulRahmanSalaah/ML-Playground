# 🚀ML-Playground: From Fundamentals to Real-World Applications
A hands-on collection of machine learning implementations ranging from algorithmic foundations to production-ready systems


## 🌟 Featured Algorithms


### 1. 🔹 Image-to-Text Generation with CNN-LSTM (EfficientNetB7)

A deep learning project that generates natural language captions from images using a **pretrained EfficientNetB7** for feature extraction and **LSTM** for sequence generation.

## Project Overview

This project demonstrates:
- CNN-based feature extraction using **EfficientNetB7 pretrained on ImageNet**
- LSTM-based sequence modeling for natural language generation
- Custom PyTorch pipeline for training and evaluation
- Flickr8k dataset with **82% BLEU-1 score**

## Key Features

- **Feature Extraction**:
  - Pretrained **EfficientNetB7** → high-quality image embeddings
- **Sequence Generation**:
  - **LSTM** decoder to generate captions word-by-word
- **Custom Pipeline**:
  - **6,000+ word vocabulary**
  - Tokenizer for text processing
  - Optimized Dataset/DataLoader classes

    <img width="813" height="555" alt="image" src="https://github.com/user-attachments/assets/47cc382b-a7b9-4425-a6d7-a435a0f0b73a" />

---
### 2. 🔹 Credit Card Fraud Detection

**Notebook:** `CreditCardTransactionDatasetOverview.ipynb`  
**Advanced machine learning system for detecting fraudulent credit card transactions using ensemble methods and neural networks.**

## Project Overview

This project demonstrates:
- Comprehensive EDA of credit card transaction patterns
- Multi-model approach with Neural Networks, Logistic Regression, and Random Forest
- Ensemble learning with Voting Classifier
- Production-ready fraud detection pipeline
- Advanced evaluation metrics for imbalanced datasets

## Key Features

- **🧠 Multi-Model Architecture**:
  - Neural Network (MLPClassifier) with 64→32 hidden layers
  - Logistic Regression with optimized threshold (0.7)
  - Random Forest with 50 estimators
  - Soft Voting Classifier ensemble
- **🔧 Production Pipeline**:
  - StandardScaler for feature normalization
  - Automated model persistence with joblib
  - Command-line training and evaluation interface
- **📊 Imbalanced Data Handling**:
  - F1-Score and PR-AUC evaluation metrics
  - SMOTE integration ready for oversampling
  - Handles extreme class imbalance (0.17% fraud rate)

## Results Summary

| Model                  | F1-Score | PR-AUC | Precision | Recall | Accuracy |
|------------------------|----------|--------|-----------|--------|----------|
| **Random Forest**      | **0.8525** | **0.8466** | 0.8966    | 0.8125 | 99.95%   |
| Neural Network (MLP)   | 0.8333   | 0.8395 | 0.8929    | 0.7812 | 99.95%   |
| Voting Classifier      | 0.7978   | 0.8370 | 0.8391    | 0.7604 | 99.93%   |
| Logistic Regression    | 0.7374   | 0.7258 | 0.7952    | 0.6875 | 99.92%   |

## 🎯 Notable Findings
- **Random Forest** achieves best overall performance with 85.25% F1-Score
- Successfully handles extreme imbalance: **96 fraud cases** out of **56,908** transactions
- High precision rates (79-89%) minimize false fraud alerts for customer experience
- Strong recall performance (68-81%) ensures most fraudulent transactions are caught

## ⚙️ Architecture

```
📁 Credit Card Fraud Detection/
├── 📓 CreditCardTransactionDatasetOverview.ipynb  # EDA & Data Analysis
├── 🐍 credit_fraud_train.py                       # Multi-model training pipeline
├── 🐍 credit_fraud_utils_eval.py                  # Comprehensive evaluation suite
├── 🐍 credit_fraud_utils_data.py                  # Data preprocessing utilities
├── 🐍 main.py                                     # Orchestration script
├── 📊 data/split/                                 # Train/test datasets
└── 💾 saved_models/                               # Serialized models + scalers
```

**Usage:**
```bash
# Train all models
python credit_fraud_train.py --train_data data/split/train.csv

# Evaluate performance  
python main.py --test_data data/split/test.csv --models_dir saved_models
```
---
### 2. 🔹 Voice Gender Classification

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
### 3. 🔹 NYC Taxi Trip Duration - EDA Summary

## 📌 Key Insights

### 1. **Data Overview**
- **1M records** | **10 features** | **No missing values**
- Avg. trip duration: **954 sec** (~16 min) | Max: **2.2M sec** (outliers present)

### 2. **Trip Duration Analysis**
- Right-skewed distribution → **Log transform applied**
- 75% of trips < **1,074 sec** (~18 min)

### 3. **Spatial Features**
- Calculated **Haversine distance** between pickup/dropoff
- Created **traffic congestion** categories (Light/Heavy)

### 4. **Temporal Patterns**
- Extracted:
  - Hour of day (peak hours)
  - Day of week (weekday/weekend)
  - Monthly trends

### 5. **Passenger Count**
- Avg: **1.67 passengers**
- Removed unrealistic values (0 or >7 passengers)

## 🔍 Notable Findings
- Strong correlation between **distance and duration**
- Significant **rush hour effects** on trip times
- **Geospatial clusters** around Manhattan

## ⚙️ Feature Engineering
1. Log-transformed target (`trip_duration`)
2. Derived:
   - Straight-line distance
   - Time-based features
   - Traffic indicators

---
### 4. 🔹 Neural Network from Scratch (MNIST Multiclass Classifier)

A fully connected, **vectorized neural network** built entirely from scratch using NumPy — no external ML libraries.

#### 🧠 Project Overview

This project demonstrates:

- Implementation of forward and backward propagation manually
- Batch training using cross-entropy loss
- Multiclass classification on the MNIST dataset (digits 0–9)
- Fast **vectorized NumPy operations** (no explicit loops)
- Clean design focused on understanding the fundamentals

#### ⚙️ Architecture

- **Input Layer:** 784 neurons (28×28 pixel images)
- **Hidden Layer 1:** 20 neurons, using `tanh` activation
- **Hidden Layer 2:** 15 neurons, using `tanh` activation
- **Output Layer:** 10 neurons, using `softmax` activation
#### 🔑 Key Features

- 🚀 Fully **vectorized** forward and backward pass (efficient and scalable)
- 🧮 Manual implementation of all key neural network steps
- 🧠 No libraries like TensorFlow or PyTorch — just NumPy
---

### 5. 🔹 Linear & Logistic Regression 

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
### 6. 🔹 K-Nearest Neighbors (KNN) Classifier

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
### 7. 🔹 Linear Regression

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









