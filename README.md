# Machine Learning (CS-433) - Project 1: Heart Disease Prediction

## Overview
This repository contains the code and resources for Project 1 of the Fall 2026 Machine Learning course (CS-433) at EPFL. The goal of this project is to apply foundational machine learning concepts to a real-world dataset to predict the risk of a person developing coronary heart disease (MICHD).
 
## Dataset
The dataset is derived from the Behavioral Risk Factor Surveillance System (BRFSS), containing health-related telephone survey data about U.S. residents. 
- The training data consists of over 300,000 individuals and 321 initial features.
- Initial exploratory data analysis (EDA) addresses missing values, filters out sparse columns (e.g., those with >95% missing data), and handles special survey codes (e.g., 7, 77, 9, 99 indicating "Don't know" or "Refused").

**Note:** The dataset files (`x_train.csv`, `y_train.csv`, `x_test.csv`) must be placed in the root directory (or a designated `data/` folder) to run the scripts.

## Project Structure
- `run.ipynb`: The main executable notebook that loads the data, performs preprocessing/feature engineering, trains the final model, and generates a submission file.
- `implementations.py`: Contains the raw, from-scratch implementations of the 6 required core machine learning algorithms.
- `helpers.py`: Contains provided helper functions (e.g., `load_csv_data`, `create_csv_submission`) as well as auxiliary functions for `implementations.py`.
- `report.pdf`: The 2-page report detailing our methodology, experiments, and results.

## Requirements and Environment
As per the project rules, **no external machine learning libraries (e.g., Scikit-Learn, Pandas, PyTorch, TensorFlow) are allowed.**

The allowed libraries are:
- Python standard libraries
- `numpy` (for all mathematical operations and data manipulation)
- `matplotlib` (strictly for data visualization and EDA)

## Implemented Methods
As part of the project requirements, the following baseline methods have been implemented from scratch in `implementations.py`:
1. `mean_squared_error_gd`: Linear regression using gradient descent.
2. `mean_squared_error_sgd`: Linear regression using stochastic gradient descent.
3. `least_squares`: Least squares regression using normal equations.
4. `ridge_regression`: Ridge regression using normal equations.
5. `logistic_regression`: Logistic regression using gradient descent.
6. `reg_logistic_regression`: Regularized logistic regression using gradient descent.

## How to Run
1. Clone this repository.
2. Ensure you have a standard Python 3 environment installed with `numpy`.
3. Place `x_train.csv`, `y_train.csv`, and `x_test.csv` in the same directory as the scripts.
4. Execute the main pipeline to generate predictions:
   ```bash
   python run.ipynb
   ```
5. The script will output a `submission_ridge_v1.csv` file formatted for the AIcrowd competition platform.

## Authors / Team Members
* Chloé Dupuis - chloe.dupuis@epfl.ch
* Lise Gillin - lise.gillin@epfl.ch
* Adrien Perrini - adrien.perrini@epfl.ch