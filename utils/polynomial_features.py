#!/usr/bin/env python

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd

def create_polynomial_features(X: pd.DataFrame, degree: int = 2):
    """
    Create polynomial features of the specified degree.
    
    :param X: DataFrame of features
    :param degree: Degree of the polynomial features (default is 2)
    :return: DataFrame of polynomial features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
    return X_poly_df

def normalize_features(X: pd.DataFrame):
    """
    Normalize the features using StandardScaler.
    
    :param X: DataFrame of features to normalize
    :return: DataFrame of normalized features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df

def select_top_features(data: pd.DataFrame, top_features: list):
    """
    Select the top N features from the data.
    
    :param data: DataFrame of features
    :param top_features: List of top feature names to select
    :return: DataFrame with top features
    """
    return data[top_features]
