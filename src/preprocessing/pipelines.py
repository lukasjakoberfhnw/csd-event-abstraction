import os
import pandas as pd
import functools as ft
import numpy as np
from tale_preprocessing import has_payload

TALE_RAW_PATH = "./data/tale-camerino/from_massimiliano/Log"
TALE_PROCESSED_PATH = "./data/tale-camerino/from_massimiliano/processed"

def load_tale_data_from_raw_files(tale_folder_location) -> pd.DataFrame:
    """
    Load the tale data from the raw files. Is used for the different preprocessings.
    """
    # main_frame = pd.DataFrame(columns=['time', 'activity', 'lifecycle', 'payload', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'robot', 'has_payload', 'run'])
    df_list = []

    folders = os.listdir(tale_folder_location)
    for i, folder in enumerate(folders):
        print(f"Processing folder {folder} ({i+1}/{len(folders)})")

        csv_files = os.path.join(tale_folder_location, folder, folder + '_0.db')

        # subfolders for the different robots
        drone_1_files = os.path.join(csv_files, 'drone_1')
        tractor_1_files = os.path.join(csv_files, 'tractor_1')
        tractor_2_files = os.path.join(csv_files, 'tractor_2')
        tractor_3_files = os.path.join(csv_files, 'tractor_3')

        # load the csv required files -- macro.csv for target variable, which ones for input?
        drone_1_macro = pd.read_csv(os.path.join(drone_1_files, "macro.csv"))
        tractor_1_macro = pd.read_csv(tractor_1_files + "/macro.csv")
        tractor_2_macro = pd.read_csv(tractor_2_files + "/macro.csv")
        tractor_3_macro = pd.read_csv(tractor_3_files + "/macro.csv")

        drone_1_odom = pd.read_csv(os.path.join(drone_1_files, "odom.csv"))
        tractor_1_odom = pd.read_csv(tractor_1_files + "/odom.csv")
        tractor_2_odom = pd.read_csv(tractor_2_files + "/odom.csv")
        tractor_3_odom = pd.read_csv(tractor_3_files + "/odom.csv")

        # add identifier to the dataframes
        drone_1_macro['robot'] = 'drone_1'
        tractor_1_macro['robot'] = 'tractor_1'
        tractor_2_macro['robot'] = 'tractor_2'
        tractor_3_macro['robot'] = 'tractor_3'

        drone_1_odom['robot'] = 'drone_1'
        tractor_1_odom['robot'] = 'tractor_1'
        tractor_2_odom['robot'] = 'tractor_2'
        tractor_3_odom['robot'] = 'tractor_3'

        mergable_dfs = [drone_1_macro, tractor_1_macro, tractor_2_macro, tractor_3_macro, drone_1_odom, tractor_1_odom, tractor_2_odom, tractor_3_odom]

        # merge the dataframes
        full_df = pd.concat(mergable_dfs, axis=0, join="outer", ignore_index=True).sort_values("time").reset_index(drop=True)
        full_df["run"] = i

        df_list.append(full_df)

    main_frame = pd.concat(df_list, ignore_index=True)
    return main_frame

def preprocess_manual_preparation(raw_data) -> pd.DataFrame:
    """
    APPROACH 3: Preprocess the data using the findings from the manual data analysis.
    """

    # fill x,y,z for rows that are coming from the macro.csv file - thus don't have x,y,z values
    filled_main_frame = raw_data.copy()
    filled_main_frame["time"] = pd.to_datetime(filled_main_frame["time"])
    filled_main_frame["x"] = filled_main_frame.groupby("robot")["x"].ffill()
    filled_main_frame["y"] = filled_main_frame.groupby("robot")["y"].ffill()
    filled_main_frame["z"] = filled_main_frame.groupby("robot")["z"].ffill()

    filled_main_frame["dz"] = filled_main_frame.groupby("robot")["z"].diff()
    filled_main_frame["dy"] = filled_main_frame.groupby("robot")["y"].diff()
    filled_main_frame["dx"] = filled_main_frame.groupby("robot")["x"].diff()

    # for each run, fill EXPLORE activity for the drone_1 robot from START to STOP
    drone_1_indeces = filled_main_frame[(filled_main_frame['robot'] == 'drone_1') & (pd.isna(filled_main_frame["activity"]))].index
    for i in range(filled_main_frame["run"].max()):
        print(f"Currently processing run {i} for propagating EXPLORE")
        explore_start_indices = filled_main_frame[(filled_main_frame['activity'] == 'EXPLORE') & (filled_main_frame["lifecycle"] == "START") & (filled_main_frame["run"] == i)].index
        explore_stop_indices = filled_main_frame[(filled_main_frame['activity'] == 'EXPLORE') & (filled_main_frame["lifecycle"] == "STOP") & (filled_main_frame["run"] == i)].index

        for j in range(len(explore_start_indices)):
            start_index = explore_start_indices[j]
            stop_index = explore_stop_indices[j]
            indices = drone_1_indeces[(drone_1_indeces >= start_index) & (drone_1_indeces <= stop_index)]
            filled_main_frame.loc[indices, 'activity'] = 'EXPLORE'

    # for each run, add feature minutes_since_start
    for i in range(filled_main_frame["run"].max()):
        # create new features minutes_since_start
        print(f"Currently processing run {i} for adding feature minutes_since_start")
        filled_main_frame.loc[filled_main_frame["run"] == i, "minutes_since_start"] = (filled_main_frame.loc[filled_main_frame["run"] == i, "time"] - filled_main_frame.loc[filled_main_frame["run"] == i, "time"].min()).dt.total_seconds() / 60

    print("Adding has_payload feature")
    filled_main_frame["has_payload"] = filled_main_frame["payload"].apply(has_payload)

    # propagate TAKEOFF until the first explore activity for each run
    for i in range(filled_main_frame["run"].max()):
        print(f"Currently processing run {i} for propagating TAKEOFF")
        if i == 5 or i == 23: # run 5 is damaged and run 23 does for soome reason not have a TAKEOFF activity
            continue
        first_takeoff = filled_main_frame[(filled_main_frame["run"] == i) & (filled_main_frame["activity"] == "TAKEOFF")].index[0]
        first_explore = filled_main_frame[(filled_main_frame["run"] == i) & (filled_main_frame["activity"] == "EXPLORE")].index[0]

        # fill all drone rows with TAKEOFF activity until the first EXPLORE activity only if the robot is drone_1
        drone_1_indeces = filled_main_frame[(filled_main_frame['robot'] == 'drone_1') & (pd.isna(filled_main_frame["activity"]))].index
        indices = drone_1_indeces[(drone_1_indeces >= first_takeoff) & (drone_1_indeces <= first_explore)]
        filled_main_frame.loc[indices, 'activity'] = 'TAKEOFF'

    return filled_main_frame

if __name__ == "__main__":
    # Generate Raw File
    tale_data = load_tale_data_from_raw_files(TALE_RAW_PATH)
    tale_data.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_raw_aggregated.csv"), index=False)

    # Preprocessings
    # tale_data = pd.read_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_raw_aggregated.csv"))
    preprocessed_3 = preprocess_manual_preparation(tale_data)
    preprocessed_3.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_preprocessed_3.csv"), index=False)

    # Load Preprocessed File
    tale_data = pd.read_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_preprocessed_3.csv"))
    print(tale_data.head())


# TODO: Clean run 5 or drop, clean run 23 or drop