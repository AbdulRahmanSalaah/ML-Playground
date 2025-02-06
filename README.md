# ML-Playground
 ML-Algorithms 🚀   A collection of machine learning algorithms implemented from scratch in Python, including linear regression, gradient descent, and more. Ideal for learning and experimenting with ML concepts.


## 📌 Table of Contents
- [Linear Regression](#linear-regression)
  
  - Implemented using **Gradient Descent**  
  - Uses **ArgumentParser** to allow flexible execution with different options  
  - Supports different preprocessing options:  
    - `0`: No preprocessing  
    - `1`: Min/Max scaling  
    - `2`: Standardization  
  - Supports different choices for training:  
    - `0`: Verify a perfect linear line (45-degree)  
    - `1`: Train with all features  
    - `2`: Train with the best feature  
    - `3`: Solve using Normal Equations  
    - `4`: Solve using Scikit-Learn  
  - Adjustable parameters:  
    - `--step_size`: Learning rate  
    - `--precision`: Requested precision  
    - `--max_iter`: Number of epochs  
    - `--dataset`: Dataset file to use  
