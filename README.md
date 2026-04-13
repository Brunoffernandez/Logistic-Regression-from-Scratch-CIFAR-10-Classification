# Logistic Regression from Scratch & CIFAR-10 Classification

##  Project Overview
This repository contains the mathematical implementation of classic Machine Learning algorithms and their application to a real-world computer vision problem. The project was originally developed as part of an academic assignment on Statistical Learning in collaboration with peers.

**My main contributions (and the focus of this repository) include:**
1. **Algorithmic Implementation:** Built Maximum Likelihood Estimation (MLE) from scratch using the second-order optimization algorithm **Newton-Raphson**.
2. **Mathematical Optimization:** Resolved convergence issues and singular matrices by implementing the **Ridge penalty (L2 Regularization)** directly into the Hessian matrix calculation.
3. **Model Evaluation:** Conducted classification analysis using stratified cross-validation, interpreting the discrepancies between *Accuracy* and *Log-Loss* metrics.

## Technologies Used
* **Language:** Python 3
* **Key Libraries:** NumPy (linear algebra), Scikit-Learn (modeling and validation), Matplotlib (data visualization)
* **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Image Classification)

## Key Findings

* **Mathematical Convergence:** The manual implementation of the Newton-Raphson method achieves fast convergence thanks to second-order information (Hessian). However, it requires a Ridge penalty to avoid non-invertible (singular) matrices in certain data spaces.
* **Accuracy vs. Log-Loss:** During the search for regularization hyperparameters ($C$), it was found that optimizing for *Log-Loss* results in more robust models. This metric not only penalizes incorrect predictions but also the model's false confidence in its predictions.

## Repository Structure
* `src/optimizers.py`: Contains the pure mathematical optimization functions (`logistic_regression_NR` and `logistic_regression_NR_penalized`).
* `src/cifar10_classification.py`: Script for image preprocessing (scaling to [0,1]) and final model fitting using `StratifiedKFold`.
