import os
import pandas as pd

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed")
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')

dtype = {'time': 'str', 'activity': 'category', 'x': 'float', 'y': 'float', 'z': 'float', 'robot': 'category', 'run': 'int', 'has_payload': 'category'}

def main():
    full_df = pd.DataFrame(columns=['time', 'activity', 'lifecycle', 'payload', 'x', 'y', 'z', 'robot', 'run', 'has_payload'])

    # use the multiple runs to predict activity
    data = pd.read_csv(os.path.join(data_path, 'full_dataset.csv'), dtype=dtype)

    # print statistics about runs
    print(data['run'].value_counts())

    # print statistics for each run and activity
    print(data.groupby(['run', 'activity']).size())


if __name__ == "__main__":
    main()