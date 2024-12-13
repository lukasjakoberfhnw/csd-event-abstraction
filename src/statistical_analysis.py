import os
import pandas as pd

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed")
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')

dtype = {'time': 'str', 'activity': 'category', 'x': 'float', 'y': 'float', 'z': 'float', 'robot': 'category', 'run': 'int', 'has_payload': 'category'}

def main():
    full_df = pd.DataFrame(columns=['time', 'activity', 'lifecycle', 'payload', 'x', 'y', 'z', 'robot', 'run', 'has_payload'])

    # use the multiple runs to predict activity
    for file in os.listdir(data_path):
        if file.endswith(".csv") and not file.startswith("16"):
            df = pd.read_csv(os.path.join(data_path, file), dtype=dtype)

            print("Loading file", file)

            # print all nan values of the dataframe
            # print(df.isna().sum())

            full_df = pd.concat([full_df, df])

    # print statistics about runs
    print(full_df['run'].value_counts())

    # print statistics for each run and activity
    print(full_df.groupby(['run', 'activity']).size())


if __name__ == "__main__":
    main()