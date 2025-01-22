import os
import pandas as pd
import functools as ft

data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
tale_XES_file_location = os.path.join(data_path, 'tale-camerino', 'MRS.xes')
tale_folder_location = os.path.join(data_path, 'tale-camerino', 'from_massimiliano', 'Log')
tale_processed_location = os.path.join(data_path, 'tale-camerino', 'from_massimiliano', 'processed')
    
def has_payload(payload):
    if payload is None or pd.isna(payload) or payload == "":
        return "no_payload"
    if "/tractor" in str(payload):
        return "/tractor" # most likely to be CUT_GRASS
    elif "name" in str(payload):
        return "name" # most likely to be CLOSTEST_TRACTOR
    elif "header: tractor" in str(payload):
        return "header: tractor" # most likely to be TRACTOR_POSITION
    elif "weed" in str(payload):
        return "weed" # most likely to be WEED_POSITION
    else:
        return "unknown"

def load_folders_and_preprocess_data():
    # get all folders in the data/tale/from_massimiliano folder
    folders = os.listdir(tale_folder_location)
    print(folders)

    # have a main dataframe to collect all the data
    main_frame = pd.DataFrame(columns=['time', 'activity', 'lifecycle', 'payload', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'robot', 'has_payload', 'run'])

    for i, folder in enumerate(folders):
        print(f"Processing folder {folder} ({i+1}/{len(folders)})")
        # only load the first 3 folders to reduce memory usage
        full_df = pd.DataFrame(columns=['time', 'activity', 'lifecycle', 'payload', 'x', 'y', 'z', 'robot'])
        # if i > 0:
        #     break

        # get the correct csv file we want to use or multiple files?
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

        # remove the first n rows until the first activity
        # first_activity_index = full_df[full_df["activity"] == "TAKEOFF"].index[0]
        # full_df = full_df.iloc[first_activity_index:, :]

        # fill activities between start and end with the same activity
        # full_df["activity"] = full_df["activity"].fillna(method='ffill')
        
        # fill activities between start and stop with the same activity based on robot
        full_df["activity"] = full_df.groupby("robot")["activity"].ffill()

        # fill the X, Y, Z values with the last known value
        full_df["x"] = full_df.groupby("robot")["x"].ffill()
        full_df["y"] = full_df.groupby("robot")["y"].ffill()
        full_df["z"] = full_df.groupby("robot")["z"].ffill()

        # calculate the difference between the current and previous X, Y, Z values
        full_df["dx"] = full_df.groupby("robot")["x"].diff()
        full_df["dy"] = full_df.groupby("robot")["y"].diff()
        full_df["dz"] = full_df.groupby("robot")["z"].diff()

        # fill the rest of the missing values with UNKNOWN
        full_df["activity"] = full_df["activity"].fillna("UNKNOWN")
        full_df["has_payload"] = full_df["payload"].apply(has_payload)

        full_df['time'] = pd.to_datetime(full_df['time'])
        full_df['activity'] = full_df['activity'].astype('category')
        full_df['has_payload'] = full_df['has_payload'].astype('category')
        full_df['robot'] = full_df['robot'].astype('category')

        full_df = full_df.drop(columns=['payload', 'lifecycle'])

        # save the dataframe to a csv file
        full_df.to_csv(os.path.join(tale_processed_location, str(i) + "_" + folder + '.csv'), index=False)
        main_frame = pd.concat([main_frame, full_df], axis=0, join="outer", ignore_index=True)
        # clear the full_df to save memory
        del full_df

    # save the main dataframe to a csv file
    main_frame.to_csv(os.path.join(tale_processed_location, 'full_dataset.csv'), index=False)

def main():
    load_folders_and_preprocess_data()

if __name__ == '__main__':
    main()