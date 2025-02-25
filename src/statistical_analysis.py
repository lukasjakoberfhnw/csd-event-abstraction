import os
import pandas as pd

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed")
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')

def main():
    # get unique values from preprocessed 4 dataset
    preprocessing_file_4_path = os.path.join(data_path, "tale_data_preprocessed_4_train.csv")
    df = pd.read_csv(preprocessing_file_4_path)

    # get unique activity for drone, robot, and shared
    print(df.head())

    # get unique activity for drone
    drone_rows = df[df['robot_drone_1'] == True]

    # get unique activity for drone
    robot_rows = df[df['robot_drone_1'] == False]

    # get unique activity for shared
    activity_shared = df['activity'].unique()

    print(drone_rows['activity'].unique())
    print(robot_rows['activity'].unique())

    

    print(activity_shared)




if __name__ == "__main__":
    main()