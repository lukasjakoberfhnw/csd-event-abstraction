from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

class ManualPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_function, train):
        super().__init__()
        self.robot_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.has_payload_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.preprocessing_function = preprocessing_function
        self.train = train

    def fit(self, X, y=None):
        """Apply preprocessing before fitting encoders."""

        # Apply the preprocessing function to generate "has_payload"
        preprocessed_data = self.preprocessing_function(X, train=self.train)

        # Fit encoders on categorical columns
        self.robot_encoder.fit(preprocessed_data[["robot"]])

        if "has_payload" in preprocessed_data.columns:
            self.has_payload_encoder.fit(preprocessed_data[["has_payload"]])
        else:
            print("Warning: 'has_payload' column is missing during fit.")

        return self  # Always return self in fit()

    def transform(self, X):
        """Apply preprocessing and encode categorical variables."""
        
        preprocessed_data = self.preprocessing_function(X, train=self.train)

        # Encode categorical columns
        preprocessed_data["robot"] = self.robot_encoder.transform(preprocessed_data[["robot"]])

        if "has_payload" in preprocessed_data.columns:
            preprocessed_data["has_payload"] = self.has_payload_encoder.transform(preprocessed_data[["has_payload"]])
        else:
            print("Warning: 'has_payload' column is missing after preprocessing.")

        # **Drop activity before returning**
        if "activity" in preprocessed_data.columns:
            preprocessed_data = preprocessed_data.drop(columns=["activity"])

        return preprocessed_data.fillna(0)  # Replace NaNs with 0
