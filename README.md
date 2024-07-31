# PyDataToolkit

![PyDataToolkit Logo](https://github.com/Fredericcelerse/PyDataToolkit/raw/main/logo.png)

PyDataToolkit is a set of Python tools designed to simplify data processing and analysis. This project includes functionalities for data loading, categorical encoding, data normalization, polynomial feature creation, and more.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

PyDataToolkit provides a comprehensive suite of tools to streamline your data science workflow. Whether you are loading data, transforming features, or preparing data for machine learning, PyDataToolkit has you covered.

## Features

- **Data Cleaning**: Clean training and test datasets from CSV files.
- **Categorical Encoding**: One-hot encode categorical features.
- **Data Normalization**: Normalize features using standard scaling.
- **Polynomial Features**: Generate polynomial features to capture non-linear relationships.
- **Correlation Analysis**: Identify and select top features correlated with the target variable.
- **Visualization**: Visualize missing data and feature correlations.

## Installation

To install PyDataToolkit, you can clone the repository and install the required dependencies:

```bash
git clone https://github.com/Fredericcelerse/PyDataToolkit.git
cd PyDataToolkit
pip install -r requirements.txt

## Usage

### Categorical Encoding
Encode categorical columns using the **encode_categorical_columns** function:
```python
from utils.categorical_encoding import encode_categorical_columns

categorical_cols = identify_categorical_columns(train_data, exclude_columns=['SalePrice'])
train_data_encoded_df = encode_categorical_columns(train_data, categorical_cols)
```

### Normalizing Data
Normalize features using the **normalize_data** function:
```python
from utils.normalization import normalize_data

train_data_scaled_df = normalize_data(train_data_encoded_df)
```

### Polynomial Features
Generate polynomial features with the **create_polynomial_features** function:
```python
from utils.polynomial_features import create_polynomial_features

X_poly_df = create_polynomial_features(X, degree=2)
```

### Visualization
Visualize missing data and feature correlations:
```python
from utils.visualization import visualize_missing_data, plot_correlation_heatmap

visualize_missing_data(train_data)
plot_correlation_heatmap(train_data, top_features)
```

## Contributing

Contributions are welcome! 
