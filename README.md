# Diabetes Detection System

A complete **Machine Learning and Deep Learning project** to predict diabetes based on patient health data. This project demonstrates **data preprocessing, handling missing values, feature scaling, class imbalance handling using SMOTE**, and building multiple predictive models with evaluation metrics.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Evaluation](#models--evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Overview
The Diabetes Detection System predicts whether a patient has diabetes using medical attributes such as Glucose, Blood Pressure, BMI, Insulin levels, and more. The project implements and compares:

- **Logistic Regression**  
- **Random Forest Classifier**  
- **Neural Network using Keras/TensorFlow**  

It also visualizes **ROC curves** to compare model performance.

---

## Dataset
- **Source:** [Pima Indians Diabetes Database - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Features:**  
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
- **Target:**  
  - Outcome (0 = No Diabetes, 1 = Diabetes)  

---

## Features
- Preprocessing: handle missing or zero values, scale features.  
- Handle **class imbalance** using **SMOTE**.  
- Compare **Logistic Regression, Random Forest, and Neural Network** models.  
- Evaluate models with **Accuracy, Precision, Recall, F1-score, ROC-AUC**, and **Confusion Matrix**.  
- Visualize **ROC curves** to identify the best model.

---

## Installation
1. Clone the repository:
git clone https://github.com/saarthakmathur/diabetes-detection.git
cd diabetes-detection

## Install dependencies:

pip install -r requirements.txt
Usage

Run the main Python script:

python diabetes_detection.py


## The script will:

Load and preprocess the dataset.

Handle class imbalance using SMOTE.

Train Logistic Regression, Random Forest, and Neural Network models.

Evaluate and print model metrics.

Plot ROC curves for all models.

## Models & Evaluation

Logistic Regression: Baseline ML model.

Random Forest: Ensemble model with class weighting.

Neural Network: Keras-based deep learning model with Dropout layers.

### Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

## Results

The project prints detailed metrics for all three models.

ROC curves allow visual comparison of model performance.

SMOTE improves model accuracy on imbalanced datasets.

## Dependencies

pandas

numpy

scikit-learn

imbalanced-learn

matplotlib

seaborn

tensorflow

### Install all dependencies with:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn tensorflow

## Author

### Saarthak Mathur