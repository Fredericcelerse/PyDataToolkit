#!/usr/bin/env python

"""
OWNER: Frédéric CELERSE
Date: 31/07/2024
Goal: This script gives a small example on how to use the functions in the utils folder
      to create new elegant features
"""

# We import the libraries we need
import pandas
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils.correlation import *
from utils.polynomial_features import *
from utils.calculate_correlations import *

# Load the data
train_data = load_processed_data('../out/processed_train_data.csv')

# Calculate the correlation matrix
corr_matrix = calculate_correlation_matrix(train_data)

# Get the absolute correlations with the target variable 'SalePrice'
top_features = get_top_correlated_features(corr_matrix, 'SalePrice', top_n=10)
print("Top features most correlated with SalePrice:")
print(top_features)

# Create a heatmap of the correlation matrix for these top features
plot_correlation_heatmap(train_data, top_features)

# Separate features and target
X = train_data.drop(columns=['SalePrice'])
y = train_data['SalePrice']

# Create polynomial features of degree 2
X_poly_df = create_polynomial_features(X, degree=2)

# Normalize the features
X_poly_scaled_df = normalize_features(X_poly_df)

# Print the number of created features
print(f"Number of created features: {X_poly_scaled_df.shape[1]}")

# Compute the correlations
correlations = calculate_correlations(X_poly_scaled_df, y)
print("Correlations with the target:")
top_correlated_features = get_top_correlated_features(correlations, top_n=10)
print(top_correlated_features)
print(f"Number of features = {len(correlations)}")

# Save the new train set with only the best 10 features
top_correlated_features.to_csv('../out/test_top_10_features.csv', index=False)

# We do the same for the test set
test_data = load_processed_data('../out/processed_test_data.csv')
poly = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()
test_poly_df = generate_polynomial_features(test_data, poly)
test_poly_scaled_df = normalize_features(test_poly_df, scaler)
test_top_10_features_df = select_top_features(test_poly_scaled_df, top_correlated_features)
print("Test DataFrame with top 10 features:\n", test_top_10_features_df.head())
test_top_10_features_df.to_csv('../out/test_top_10_features.csv', index=False)
