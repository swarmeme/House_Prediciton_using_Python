# California Housing Price Prediction

![California Housing](california_housing.jpg)

This project aims to predict housing prices in California based on various features using machine learning techniques. It includes data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Introduction

In the field of real estate, predicting housing prices accurately is crucial for both buyers and sellers. This project employs various machine learning models to predict housing prices in California based on features such as total rooms, total bedrooms, population, and more.

## Project Overview

The project is divided into several key phases:

- **Data Loading:** The California housing dataset is loaded from a CSV file into a Pandas DataFrame.

- **Data Preprocessing:** Data is cleaned by handling missing values, and the dataset is split into features (X) and the target variable (y).

- **Data Transformation:** Skewed numeric features are transformed using logarithmic scaling.

- **Categorical Feature Encoding:** The categorical feature "ocean_proximity" is one-hot encoded.

- **Data Visualization:** Visualizations are created to explore the data and understand feature relationships.

- **Feature Engineering:** New features, "bedroom_ratio" and "household_ratio," are engineered.

- **Model Training:** Multiple regression models, including Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor, are trained.

- **Hyperparameter Tuning:** Hyperparameter tuning is performed for Random Forest and Gradient Boosting models using GridSearchCV.

- **Model Evaluation:** Models are evaluated using the R-squared (R2) score and mean squared error (MSE).

## Dataset

The dataset used for this project is the [California Housing Prices dataset](https://github.com/yourusername/your-repo/raw/main/data/housing.csv). It contains various attributes for housing districts in California and their median house values.

## Installation

To run this project, you'll need Python and the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
