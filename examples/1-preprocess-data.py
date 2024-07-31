#!/usr/bin/env python

"""
OWNER: Frédéric CELERSE
Date: 31/07/2024
Goal: This script gives a small example on how to use the functions in the utils folder
      to preprocess efficiently the data
"""

# We import the libraries we need
import pandas
from utils.clean import *
from utils.encoding import *
from utils.normalization import *
from utils.visualization import *


# First, we load the data
train_data = pd.read_csv('../databases/train.csv')
test_data = pd.read_csv('../databases/test.csv')

# We show the data
print(train_data.head())
print(train_data.describe())

# Then, we first identify the missing data
missing_train_data = check_missing_data(train_data)
missing_test_data = check_missing_data(train_data)
missing_data = pd.concat([missing_train_data, missing_test_data])
data_cleaned = clean_missing_data(missing_data)
print("\nList of mssing data:")
print(missing_data[missing_data > 0])
print(f"Number of features affected by the missing data: {len(data_cleaned)}/{len(data.iloc[0])-1}")

# We can optionnaly visualize the missing data
visualize_missing_data(train_data)
visualize_missing_data(test_data)

# We drop columns with more than 50% missing data
train_data = drop_columns_with_missing_data(train_data)
test_data = drop_columns_with_missing_data(test_data)
print("Number of columns after dropping: ", len(train_data.columns))

# We separate target column (target being here the 'SalePrice' column)
train_data, target = separate_target_column(train_data, 'SalePrice')

# Impute missing values in train data
train_data = impute_missing_values(train_data)

# Impute missing values in test data
test_data = impute_missing_values(test_data)

print("Check for missing data after imputation in train data:")
print(train_data.isnull().sum())
print("Check for missing data after imputation in test data:")
print(test_data.isnull().sum())

# We saw that we need to drop manually the 'FireplaceQu' column from the train_data
train_data = train_data.drop(columns=['FireplaceQu'])

# Check if the remaining features are the same in both dataframes
common_features = train_data.columns.intersection(test_data.columns)
print("Common features between train_data and test_data (excluding 'SalePrice'):", common_features)
print("Number of columns in train_data:", len(train_data.columns))
print("Number of columns in test_data:", len(test_data.columns))
print("Number of common features:", len(common_features))

# We add again 'SalePrice' to the data table
train_data['SalePrice'] = target
# We save the new training data into a new file
train_data.to_csv("../out/new_training.csv")
test_data.to_csv("../out/new_test.csv")

# Identify categorical columns, excluding 'SalePrice'
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'SalePrice']

# Encode categorical columns
train_data_encoded_df = encode_categorical_columns(train_data, categorical_cols)
test_data_encoded_df = encode_categorical_columns(test_data, categorical_cols)

# Normalize data
train_data_scaled_df = normalize_data(train_data_encoded_df)
test_data_scaled_df = normalize_data(test_data_encoded_df)

# Add 'SalePrice' back to the training data
train_data_scaled_df['SalePrice'] = train_data['SalePrice'].values

# Visualize missing data
visualize_missing_data(train_data)
visualize_missing_data(test_data)

# Save the processed data
train_data_scaled_df.to_csv('../out/processed_train_data.csv', index=False)
test_data_scaled_df.to_csv('../out/processed_test_data.csv', index=False)
