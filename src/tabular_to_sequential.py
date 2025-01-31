import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import LabelEncoder

def to_sequential(df: pd.DataFrame, out_name: str):
    feature_columns = df.columns
    feature_columns.delete(feature_columns.get_loc("activity"))  # Remove the activity column

    # Example: Assume we have columns ['timestamp', 'sensor1', 'sensor2', ..., 'activity']
    feature_columns = feature_columns  # Adjust to your dataset
    label_column = 'activity'

    # Convert categorical activity labels to numbers with label encoding from sklearn
    labelEncoder = LabelEncoder()
    df[label_column] = labelEncoder.fit_transform(df[label_column]) 

    # Convert True/False columns to 1/0
    df = df.astype({col: int for col in df.columns if df[col].dtype == bool})

    print("Data shape:", df.shape)
    print("Data columns:", df.columns)
    print(df.head())

    print("NaN values in data:", df.isnull().sum().sum())

    # Define sequence length
    sequence_length = 50  # Adjust as needed

    # Convert tabular data into sequences
    X_sequences = []
    Y_sequences = []

    for i in range(len(df) - sequence_length):
        X_sequences.append(df[feature_columns].iloc[i:i+sequence_length].values)  # Input features
        Y_sequences.append(df[label_column].iloc[i:i+sequence_length].values)  # Activity labels

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)  # Shape: (num_samples, seq_len, num_features)
    Y_tensor = torch.tensor(np.array(Y_sequences), dtype=torch.long)  # Shape: (num_samples, seq_len)

    # Print final shape
    print("Input shape (X):", X_tensor.shape)  # (num_samples, sequence_length, num_features)
    print("Target shape (Y):", Y_tensor.shape)  # (num_samples, sequence_length)

    # Save tensors for future training
    # torch.save((X_tensor, Y_tensor), out_name)
    return X_tensor, Y_tensor, labelEncoder
