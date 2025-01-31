import pandas as pd
import os
from tabular_to_sequential import to_sequential

# Load preprocessing 4 dataset
preprocessing_file_4_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed", "tale_data_preprocessed_3_5_train.csv")
df = pd.read_csv(preprocessing_file_4_path)
print(len(df))

df.dropna(inplace=True)

print(df.shape)

# get index of the end of the first run
first_run_index = df[df['run'] == 0].index[-1]
second_run_index = df[df['run'] == 1].index[-1]

# df = df.iloc[:20000]
df_train = df.iloc[0:first_run_index]
df_test = df.iloc[first_run_index:second_run_index]

print(len(df_train))
print(len(df_test))

test_df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, pd.NA, pd.NA]], columns=['A', 'B', 'C'])

print(test_df)
print(test_df.isnull().sum().sum())

test_df.dropna(inplace=True)

print(test_df)

X_train, y_train, labelEncoder_train = to_sequential(df_train, "sequential_data.pth")
X_test, y_test, labelEncoder_test = to_sequential(df_test, "sequential_data_test.pth")

# Check the shape of loaded tensors
print(X_train.shape)
print(y_train.shape)

