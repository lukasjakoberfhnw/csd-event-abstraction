import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import LabelEncoder

def to_sequential(df: pd.DataFrame, out_name: str, recreate: bool = False):
    if os.path.exists(out_name) and not recreate:
        print("Loading existing tensors...")
        X_tensor, Y_tensor = torch.load(out_name)
        labelEncoder = torch.load(out_name + "_labelEncoder")
        return X_tensor, Y_tensor, labelEncoder
    
    feature_columns = df.columns
    feature_columns.delete(feature_columns.get_loc("activity"))  # Remove the activity column

    # Example: Assume we have columns ['timestamp', 'sensor1', 'sensor2', ..., 'activity']
    feature_columns = feature_columns  # Adjust to your dataset
    label_column = 'activity'

    # Convert categorical activity labels to numbers with label encoding from sklearn
    labelEncoder = LabelEncoder()
    df[label_column] = labelEncoder.fit_transform(df[label_column]) 

    # Store label encoding for future reference to file
    labelEncoder_file = out_name + "_labelEncoder"
    torch.save(labelEncoder, labelEncoder_file)

    # Convert True/False columns to 1/0
    df = df.astype({col: int for col in df.columns if df[col].dtype == bool})

    print("Data shape:", df.shape)
    print("Data columns:", df.columns)
    print(df.head())

    print("NaN values in data:", df.isnull().sum().sum())

    # Define sequence length
    sequence_length = 50 # length of sequence
    stride = 10 # how much we slide in the sliding window approach :)

    # Convert tabular data into sequences
    X_sequences = []
    Y_sequences = []

    for i in range(0, len(df) - sequence_length, stride):
        X_sequences.append(df[feature_columns].iloc[i:i+sequence_length].values)  # Input features
        Y_sequences.append(df[label_column].iloc[i:i+sequence_length].values)  # Activity labels

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)  # Shape: (num_samples, seq_len, num_features)
    Y_tensor = torch.tensor(np.array(Y_sequences), dtype=torch.long)  # Shape: (num_samples, seq_len)

    # Print final shape
    print("Input shape (X):", X_tensor.shape)  # (num_samples, sequence_length, num_features)
    print("Target shape (Y):", Y_tensor.shape)  # (num_samples, sequence_length)

    # Save tensors for future training
    torch.save((X_tensor, Y_tensor), out_name)
    return X_tensor, Y_tensor, labelEncoder

def main():
    # load created tensors for training and testing
    X_train, y_train = torch.load("sequential_data.pth")
    X_test, y_test = torch.load("sequential_data_test.pth")

    # Load label encoders
    # labelEncoder_train = torch.load("sequential_data.pth_labelEncoder")
    # labelEncoder_test = torch.load("sequential_data_test.pth_labelEncoder")

    # Check the shape of loaded tensors
    print("Loaded Input Shape:", X_train.shape)  # (num_samples, sequence_length, num_features)
    print("Loaded Target Shape:", y_train.shape)  # (num_samples, sequence_length)
    print("Loaded Input Shape:", X_test.shape)  # (num_samples, sequence_length, num_features)
    print("Loaded Target Shape:", y_test.shape)  # (num_samples, sequence_length)


    # Check unique classes
    print("Unique classes:", torch.unique(y_train))
    print("Unique classes:", torch.unique(y_test))

    # Unique classes are wrong... maybe because of the shape of the target tensor
    # We need to flatten the target tensor to get the unique classes
    print("Unique classes:", torch.unique(y_train.view(-1)))
    print("Unique classes:", torch.unique(y_test.view(-1)))

    # Load preprocessing 4 dataset
    preprocessing_file_4_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed", "tale_data_preprocessed_4_train.csv")
    df = pd.read_csv(preprocessing_file_4_path)
    print(len(df))

    df.dropna(inplace=True)

    # get index of the end of the first run
    first_run_index = df[df['run'] == 0].index[-1]
    
    supposed = 12
    for i in range(df["run"].nunique()):
        print(f"Run {i}:", df[df['run'] == i].shape)
        # print(f"Run {i}:", df[df['run'] == i]["activity"].value_counts())
        #print(f"Run {i}:", df[df['run'] == i]["activity"].nunique())
        if(df[df['run'] == i]["activity"].nunique() != supposed):
            print(f"Run {i}:", df[df['run'] == i]["activity"].nunique())


    # Check classes in labelEncoder
    # print("Classes in labelEncoder_train:", labelEncoder_train.classes_)
    # print("Classes in labelEncoder_test:", labelEncoder_test.classes_)

if __name__ == "__main__":
    main()