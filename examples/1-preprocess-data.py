#!/usr/bin/env python

"""
OWNER: Frédéric CELERSE
Date: 31/07/2024
Goal: This script gives a small example on how to use the functions in the utils folder
      to preprocess efficiently the data
"""

# We import the libraries we need
import pandas

# First, we load the data
train_data = pd.read_csv('../databases/train.csv')
test_data = pd.read_csv('../databases/test.csv')

# We show the data
print(train_data.head())
print(train_data.describe())

# The, we first clean our data
missing_train_data = check_missing_data(train_data)
missing_test_data = check_missing_data(train_data)
missing_data = pd.concat([missing_train_data, missing_test_data])
