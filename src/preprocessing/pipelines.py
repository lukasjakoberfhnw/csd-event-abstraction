import os
import pandas as pd
import functools as ft
import numpy as np
from tale_preprocessing import has_payload

TALE_RAW_PATH = "./data/tale-camerino/from_massimiliano/Log"
TALE_PROCESSED_PATH = "./data/tale-camerino/from_massimiliano/processed"

def load_tale_data_from_raw_files(tale_folder_location: str) -> pd.DataFrame:
    """
    Load the tale data from the raw files. Is used for the different preprocessings.

    :param str tale_folder_location: location of the tale folder (/Log directory)
    :return: loaded tale data
    :rtype: pd.DataFrame
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

def preprocess_ffill_xyz(raw_data: pd.DataFrame, train:bool) -> pd.DataFrame:
    """
        APPROACH 1: Preprocess the data using forward fill for x, y, z values.
    
        :param pd.DataFrame raw_data: loaded tale data from the raw files as a DataFrame
        :param bool train: whether the data is training data or not
        :return: preprocessed data
        :rtype: pd.DataFrame
    """
    print("Preprocessing data using ffill for x, y, z values")

    # just ffill the values for x, y, z
    filled_main_frame = raw_data.copy()
    filled_main_frame["time"] = pd.to_datetime(filled_main_frame["time"])
    filled_main_frame["x"] = filled_main_frame.groupby("robot")["x"].ffill()
    filled_main_frame["y"] = filled_main_frame.groupby("robot")["y"].ffill()
    filled_main_frame["z"] = filled_main_frame.groupby("robot")["z"].ffill()

    # remove the columns that are not needed
    filled_main_frame.drop(columns=['time', 'lifecycle', 'payload', 'run'], inplace=True)

    if train:
        filled_main_frame["activity"] = filled_main_frame["activity"].fillna("IDLE")
    else:
        filled_main_frame = filled_main_frame.drop(columns=["activity"])

    # one hot encode the robot column
    filled_main_frame = pd.get_dummies(filled_main_frame, columns=['robot'])

    return filled_main_frame

def preprocess_ffill_activities(raw_data: pd.DataFrame, train: bool) -> pd.DataFrame:
    """
        APPROACH 2: Preprocess the data using forward fill for forward fill for activities and x, y, z values.

        :param pd.DataFrame raw_data: loaded tale data from the raw files as a DataFrame
        :param bool train: whether the data is training data or not
        :return: preprocessed data
        :rtype: pd.DataFrame
    """
    print("Preprocessing data using forward fill for activities and x, y, z values")

    # just ffill the values for x, y, z and activity
    filled_main_frame = raw_data.copy()
    filled_main_frame["time"] = pd.to_datetime(filled_main_frame["time"])
    filled_main_frame["x"] = filled_main_frame.groupby("robot")["x"].ffill()
    filled_main_frame["y"] = filled_main_frame.groupby("robot")["y"].ffill()
    filled_main_frame["z"] = filled_main_frame.groupby("robot")["z"].ffill()
    filled_main_frame["activity"] = filled_main_frame.groupby("robot")["activity"].ffill()

    # remove the columns that are not needed
    filled_main_frame.drop(columns=['time', 'lifecycle', 'payload', 'run'], inplace=True)

    if train:
        filled_main_frame["activity"] = filled_main_frame["activity"].fillna("IDLE")
    else:
        filled_main_frame = filled_main_frame.drop(columns=["activity"])

    # one hot encode the robot column
    filled_main_frame = pd.get_dummies(filled_main_frame, columns=['robot'])

    return filled_main_frame

def preprocess_massimiliano(tale_folder_location: str, train: True) -> pd.DataFrame:
    """
        APPROACH 3: Preprocess the data similarly to how Massimiliano preprocessed them.
        Inspired by: https://github.com/strong-ms/crf-activity-recognition/blob/main/preprocessing.py
        Currently runs too long... Needs different approach.
    """

    print("Preprocessing data using Massimiliano's approach")

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

        drone_1_odom['activity'] = None
        tractor_1_odom['activity'] = None
        tractor_2_odom['activity'] = None
        tractor_3_odom['activity'] = None

        odoms = [drone_1_odom, tractor_1_odom, tractor_2_odom, tractor_3_odom]
        macros = [drone_1_macro, tractor_1_macro, tractor_2_macro, tractor_3_macro]

        for odom, macro in zip(odoms, macros):
        # Loop through macro_df to find activity time ranges
            for macro_odom_i in range(len(macro) - 1):
                row = macro.iloc[macro_odom_i]
                next_row = macro.iloc[macro_odom_i + 1]

                # Check if this row marks the start and the next row marks the end of an activity
                if row['lifecycle'] == 'START' and next_row['lifecycle'] == 'STOP' and row['activity'] == next_row['activity']:
                    start_time = row['time']
                    end_time = next_row['time']
                    activity = row['activity']

                    # Assign activity only to rows within the start and end time interval in odom_df
                    odom.loc[(odom['time'] >= start_time) & (odom['time'] <= end_time), 'activity'] = activity

            # Set "IDLE" as the default activity for rows without any assigned activity
            odom['activity'] = odom['activity'].fillna('IDLE')

        mergable_dfs = [drone_1_macro, tractor_1_macro, tractor_2_macro, tractor_3_macro, drone_1_odom, tractor_1_odom, tractor_2_odom, tractor_3_odom]
        full_df = pd.concat(mergable_dfs, axis=0, join="outer", ignore_index=True).sort_values("time").reset_index(drop=True)
        full_df["run"] = i

        df_list.append(full_df)

    main_frame = pd.concat(df_list, ignore_index=True)

    main_frame = main_frame.drop(columns=['time', 'lifecycle', 'payload'])
    main_frame = pd.get_dummies(main_frame, columns=['robot'])

    return main_frame

def preprocess_enhanced_massimiliano(tale_folder_location: str, train: True) -> pd.DataFrame:
    """
        APPROACH 3.5: Preprocess the data similarly to how Massimiliano preprocessed them, but also enhancing it with dx, dy, dz, payload 
        Inspired by: https://github.com/strong-ms/crf-activity-recognition/blob/main/training.py
    """
    print("Preprocessing data using enhanced Massimiliano's approach")

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

        drone_1_odom['activity'] = None
        tractor_1_odom['activity'] = None
        tractor_2_odom['activity'] = None
        tractor_3_odom['activity'] = None

        odoms = [drone_1_odom, tractor_1_odom, tractor_2_odom, tractor_3_odom]
        macros = [drone_1_macro, tractor_1_macro, tractor_2_macro, tractor_3_macro]

        for odom, macro in zip(odoms, macros):
        # Loop through macro_df to find activity time ranges
            for macro_odom_i in range(len(macro) - 1):
                row = macro.iloc[macro_odom_i]
                next_row = macro.iloc[macro_odom_i + 1]

                # Check if this row marks the start and the next row marks the end of an activity
                if row['lifecycle'] == 'START' and next_row['lifecycle'] == 'STOP' and row['activity'] == next_row['activity']:
                    start_time = row['time']
                    end_time = next_row['time']
                    activity = row['activity']

                    # Assign activity only to rows within the start and end time interval in odom_df
                    odom.loc[(odom['time'] >= start_time) & (odom['time'] <= end_time), 'activity'] = activity

            # Set "IDLE" as the default activity for rows without any assigned activity
            odom['activity'] = odom['activity'].fillna('IDLE')

        mergable_dfs = [drone_1_macro, tractor_1_macro, tractor_2_macro, tractor_3_macro, drone_1_odom, tractor_1_odom, tractor_2_odom, tractor_3_odom]
        full_df = pd.concat(mergable_dfs, axis=0, join="outer", ignore_index=True).sort_values("time").reset_index(drop=True)
        full_df["run"] = i

        df_list.append(full_df)

    main_frame = pd.concat(df_list, ignore_index=True)

    main_frame["dx"] = main_frame.groupby("robot")["x"].diff()
    main_frame["dy"] = main_frame.groupby("robot")["y"].diff()
    main_frame["dz"] = main_frame.groupby("robot")["z"].diff()

    main_frame["has_payload"] = main_frame["payload"].apply(has_payload)

    main_frame = main_frame.drop(columns=['time', 'lifecycle', 'payload']) # 'run' -- keep run and time in for now to test the crf_prototype
    main_frame = pd.get_dummies(main_frame, columns=['robot', 'has_payload'])

    return main_frame

def preprocess_manual_preparation(raw_data: pd.DataFrame, train: bool) -> pd.DataFrame:
    """
    APPROACH 4: Preprocess the data using the findings from the manual data analysis.

    :param pd.DataFrame raw_data: loaded tale data from the raw files as a DataFrame
    :param bool train: whether the data is training data or not
    :return: preprocessed data
    :rtype: pd.DataFrame

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

    # for each run, fill EXPLORE activity for the drone_1 robot from START to STOP only if it's the preprocessing for training data
    if train:
        drone_1_indeces = filled_main_frame[(filled_main_frame['robot'] == 'drone_1') & (pd.isna(filled_main_frame["activity"]))].index
        for i in range(filled_main_frame["run"].max() + 1):
            print(f"Currently processing run {i} for propagating EXPLORE")
            explore_start_indices = filled_main_frame[(filled_main_frame['activity'] == 'EXPLORE') & (filled_main_frame["lifecycle"] == "START") & (filled_main_frame["run"] == i)].index
            explore_stop_indices = filled_main_frame[(filled_main_frame['activity'] == 'EXPLORE') & (filled_main_frame["lifecycle"] == "STOP") & (filled_main_frame["run"] == i)].index

            # print(explore_start_indices)
            # print(explore_stop_indices)

            for j in range(len(explore_start_indices)):
                start_index = explore_start_indices[j]
                stop_index = explore_stop_indices[j]
                indices = drone_1_indeces[(drone_1_indeces >= start_index) & (drone_1_indeces <= stop_index)]
                filled_main_frame.loc[indices, 'activity'] = 'EXPLORE'

    # for each run, fill MOVE and CUT_GRASS activity for the tractors robot from START to STOP only if it's the preprocessing for training data
    if train:
        for tractor_id in range(1, 4): # 3 tractors
            tractor_indeces = filled_main_frame[(filled_main_frame['robot'] == 'tractor_' + str(tractor_id)) & (pd.isna(filled_main_frame["activity"]))].index
            for i in range(filled_main_frame["run"].max() + 1):
                if i == 5 or i == 23: # run 5 is damaged and run 23 does for soome reason not have a TAKEOFF activity
                    continue
                print(f"Currently processing run {i} for propagating MOVE")
                move_start_indices = filled_main_frame[(filled_main_frame['activity'] == 'MOVE') & (filled_main_frame["lifecycle"] == "START") & (filled_main_frame["run"] == i) & (filled_main_frame["robot"] == "tractor_" + str(tractor_id))].index
                move_stop_indices = filled_main_frame[(filled_main_frame['activity'] == 'MOVE') & (filled_main_frame["lifecycle"] == "STOP") & (filled_main_frame["run"] == i) & (filled_main_frame["robot"] == "tractor_" + str(tractor_id))].index
                # print(move_start_indices)
                # print(move_stop_indices)
                try:
                    print("Trying to fill MOVE activity for robot tractor_" + str(tractor_id))
                    for j in range(len(move_start_indices)):
                        start_index = move_start_indices[j]
                        stop_index = move_stop_indices[j]
                        indices = tractor_indeces[(tractor_indeces >= start_index) & (tractor_indeces <= stop_index)]
                        filled_main_frame.loc[indices, 'activity'] = 'MOVE'
                except IndexError:
                    print(f"Run {i} fails at MOVE activity: {len(move_start_indices)} {len(move_stop_indices)}")

                # CUT_GRASS finishes so fast in the simulation, it's not really propagating anyways...
                print(f"Currently processing run {i} for propagating CUT_GRASS")
                cut_grass_start_indices = filled_main_frame[(filled_main_frame['activity'] == 'CUT_GRASS') & (filled_main_frame["lifecycle"] == "START") & (filled_main_frame["run"] == i) & (filled_main_frame["robot"] == "tractor_" + str(tractor_id))].index
                cut_grass_stop_indices = filled_main_frame[(filled_main_frame['activity'] == 'CUT_GRASS') & (filled_main_frame["lifecycle"] == "STOP") & (filled_main_frame["run"] == i) & (filled_main_frame["robot"] == "tractor_" + str(tractor_id))].index

                try:
                    for j in range(len(cut_grass_start_indices)):
                        start_index = cut_grass_start_indices[j]
                        stop_index = cut_grass_stop_indices[j]
                        indices = tractor_indeces[(tractor_indeces >= start_index) & (tractor_indeces <= stop_index)]
                        filled_main_frame.loc[indices, 'activity'] = 'CUT_GRASS'
                except IndexError:
                    print(f"Run {i} fails at CUT_GRASS activity: {len(cut_grass_start_indices)} {len(cut_grass_stop_indices)}")

    # for each run, add feature minutes_since_start
    for i in range(filled_main_frame["run"].max() + 1):
        # create new features minutes_since_start
        print(f"Currently processing run {i} for adding feature minutes_since_start")
        filled_main_frame.loc[filled_main_frame["run"] == i, "minutes_since_start"] = (filled_main_frame.loc[filled_main_frame["run"] == i, "time"] - filled_main_frame.loc[filled_main_frame["run"] == i, "time"].min()).dt.total_seconds() / 60

    print("Adding has_payload feature")
    filled_main_frame["has_payload"] = filled_main_frame["payload"].apply(has_payload)

    # propagate TAKEOFF until the first explore activity for each run only if it's the preprocessing for training data
    if train:
        # fails because there is no run...
        for i in range(filled_main_frame["run"].max() + 1): # +1 because the runs are 0 indexed
            print(f"Currently processing run {i} for propagating TAKEOFF")
            if i == 5 or i == 23: # run 5 is damaged and run 23 does for soome reason not have a TAKEOFF activity
                continue

            try:
                first_takeoff = filled_main_frame[(filled_main_frame["run"] == i) & (filled_main_frame["activity"] == "TAKEOFF")].index[0]
                first_explore = filled_main_frame[(filled_main_frame["run"] == i) & (filled_main_frame["activity"] == "EXPLORE")].index[0]

                # fill all drone rows with TAKEOFF activity until the first EXPLORE activity only if the robot is drone_1
                drone_1_indeces = filled_main_frame[(filled_main_frame['robot'] == 'drone_1') & (pd.isna(filled_main_frame["activity"]))].index
                indices = drone_1_indeces[(drone_1_indeces >= first_takeoff) & (drone_1_indeces <= first_explore)]
                filled_main_frame.loc[indices, 'activity'] = 'TAKEOFF'
            except IndexError:
                print(f"Run {i} does not have a TAKEOFF or EXPLORE activity")

            try:
                land_start_indices = filled_main_frame[(filled_main_frame['activity'] == 'LAND') & (filled_main_frame["lifecycle"] == "START") & (filled_main_frame["run"] == i)].index
                land_end_indices = filled_main_frame[(filled_main_frame['activity'] == 'LAND') & (filled_main_frame["lifecycle"] == "STOP") & (filled_main_frame["run"] == i)].index

                for j in range(len(land_start_indices)):
                    start_index = land_start_indices[j]
                    stop_index = land_end_indices[j]
                    indices = drone_1_indeces[(drone_1_indeces >= start_index) & (drone_1_indeces <= stop_index)]
                    filled_main_frame.loc[indices, 'activity'] = 'LAND'
            except IndexError:
                print(f"Run {i} does not have correct LAND activities {len(land_start_indices)} {len(land_end_indices)}")

    # remove all columns that are not needed
    filled_main_frame.drop(columns=['time', 'lifecycle', 'payload', 'x', 'y', 'z','run'], inplace=True)

    if train:
        filled_main_frame["activity"] = filled_main_frame["activity"].fillna("IDLE")
    else:
        filled_main_frame = filled_main_frame.drop(columns=["activity"])

    # one hot encode the robot and has_payload columns
    filled_main_frame = pd.get_dummies(filled_main_frame, columns=['robot', 'has_payload'])

    # remove corrupted runs (5, 23)

    return filled_main_frame

if __name__ == "__main__":
    # Generate Raw File
    # tale_data = load_tale_data_from_raw_files(TALE_RAW_PATH)
    # tale_data.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_raw_aggregated.csv"), index=False)

    # last_run_data = tale_data[tale_data['run'] == tale_data['run'].max()]
    # print(last_run_data.head())
    # last_run_data.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_last_run.csv"), index=False)
    # preprocess_last_run = preprocess_manual_preparation(last_run_data, train=True)
    # print(preprocess_last_run.head())
    # preprocess_last_run.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_last_prep_4_train.csv"), index=False)

    # Preprocessings
    tale_data = pd.read_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_raw_aggregated.csv"))

    # preprocessed_1 = preprocess_ffill_xyz(tale_data, train=True)
    # print(preprocessed_1.head())
    # preprocessed_1.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_preprocessed_1_train.csv"), index=False)

    # preprocessed_2 = preprocess_ffill_activities(tale_data, train=True)
    # print(preprocessed_2.head())
    # preprocessed_2.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_preprocessed_2_train.csv"), index=False)

    # preprocessed_3 = preprocess_massimiliano(TALE_RAW_PATH, train=True)
    # print(preprocessed_3.head())
    # preprocessed_3.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_preprocessed_3_train.csv"), index=False)

    preprocessed_3_5 = preprocess_enhanced_massimiliano(TALE_RAW_PATH, train=True)
    print(preprocessed_3_5.head())
    preprocessed_3_5.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_preprocessed_3_5_train.csv"), index=False)

    # preprocessed_4 = preprocess_manual_preparation(tale_data, train=True)
    # print(preprocessed_4.head())
    # preprocessed_4.to_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_preprocessed_4_train.csv"), index=False)

    # Load Preprocessed File
    # tale_data = pd.read_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_preprocessed_4_train.csv"))
    # print(tale_data.head())


# TODO: Clean run 5 or drop, clean run 23 or drop