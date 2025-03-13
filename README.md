# Lung Cancer Detection Project

## Introduction
This project aims to develop a machine learning system for early lung cancer detection using patient data such as age, smoking habits, and clinical symptoms. The goal is to classify patients into "YES" (likely to have cancer) or "NO" (unlikely) based on their medical history.

## Problem
Lung cancer is a leading cause of death worldwide. Early detection is crucial for improving survival rates, but current methods can be resource-intensive. This project seeks to create a simple, data-driven tool for preliminary screening.

## Key Steps
1. **Data Preprocessing**: Balancing the dataset with **SMOTE** and reducing noise using **PCA**.
2. **Feature Engineering**: Creating interaction features (e.g., age * smoking) to improve model performance.
3. **Model Training**: Using **Random Forest**, **SVM**, and **XGBoost** with hyperparameter tuning and 10-fold cross-validation.
4. **Evaluation**: Assessing models with metrics like accuracy, precision, recall, and F1-score.

## Datasets
- **Dataset 1**: 308 high-quality records.
- **Dataset 2**: 3000 noisier records, used to test model robustness.

## Results
For detailed results and analysis, refer to the **Final Report** included in this repository.
