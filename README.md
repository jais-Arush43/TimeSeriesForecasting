# Bitcoin Price Forecasting Project

This project aims to forecast Bitcoin prices using various Recurrent Neural Network (RNN) architectures. It compares the performance of different models, including a standard RNN, an Alpha-RNN, GRU, and LSTM.

## Project Overview

The project uses historical Bitcoin price data to train and evaluate different RNN models. It includes data preprocessing, model building, training, and evaluation steps.

## Features

- Data preprocessing and normalization
- Implementation of multiple RNN architectures:
  - Standard RNN
  - Alpha-RNN (custom implementation)
  - GRU (Gated Recurrent Unit)
  - LSTM (Long Short-Term Memory)
- Hyperparameter tuning using GridSearchCV
- Model training with early stopping
- Performance comparison of different models
- Visualization of predictions and errors

## Requirements

- Python 
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels

## Usage

1. Ensure all required libraries are installed.
2. Load the Bitcoin price data (CSV file) into the project.
3. Run the Jupyter notebook or Python script to execute the entire pipeline.

## Key Components

- Data loading and preprocessing
- Feature engineering (creating lagged features)
- Model definition and compilation
- Cross-validation and hyperparameter tuning
- Model training and evaluation
- Results visualization

## Models

The project compares four different RNN architectures:
1. Simple RNN
2. Alpha-RNN (a custom RNN implementation)
3. GRU
4. LSTM

## Results

The models are compared based on their Mean Squared Error (MSE) on both training and test datasets. Visualizations are provided to compare the predicted values against the observed values, as well as the prediction errors for each model.


