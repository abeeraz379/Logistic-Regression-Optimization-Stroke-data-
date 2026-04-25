# Logistic Regression Pipeline for Stroke Prediction

## Overview
This project focuses on building a machine learning pipeline for stroke prediction using the `countstroke0` dataset. The dataset is a highly imbalanced binary classification problem, where the goal is to study how different Logistic Regression parameters affect the model’s performance.

The main idea of the project is to take the data through a full machine learning workflow, including loading, inspecting, cleaning, preprocessing, model building, evaluation, and hyperparameter tuning. Logistic Regression was used as the base model, and the pipeline was designed to make the process structured, reproducible, and suitable for finding the best parameter values.

## Project Workflow
- Loaded and inspected the dataset.
- Cleaned the data and handled missing values.
- Preprocessed the features using imputation, encoding, and scaling.
- Built a Logistic Regression pipeline.
- Evaluated the model using classification metrics and confusion matrices.
- Tuned six parameters with GridSearchCV to find the best overall result.

## Challenges Addressed
- Dealt with class imbalance in the target variable.
- Resolved solver and penalty compatibility issues in Logistic Regression.
- Used cross-validation to search for the best hyperparameters.
- Focused on improving the overall model performance through parameter tuning.

## Hyperparameters Tuned
- `C`
- `solver`
- `penalty`
- `max_iter`
- `class_weight`
- `l1_ratio`

## Purpose of the Project
The purpose of this project is to demonstrate how changing six Logistic Regression parameters affects model performance and how to choose the best values to get the best overall result for the classification task.
