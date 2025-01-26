from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .tale_preprocessing import has_payload

class ManualPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.robot_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, X, y=None):
        # Fit encoders on categorical columns
        self.robot_encoder.fit(X[["robot"]])
        return self

    def transform(self, X):
        """Preprocess the data using manual feature engineering."""
        filled_main_frame = X.copy()

        # Convert time column
        filled_main_frame["time"] = pd.to_datetime(filled_main_frame["time"])

        # Forward fill missing coordinates
        filled_main_frame["x"] = filled_main_frame.groupby("robot")["x"].ffill()
        filled_main_frame["y"] = filled_main_frame.groupby("robot")["y"].ffill()
        filled_main_frame["z"] = filled_main_frame.groupby("robot")["z"].ffill()

        # Compute changes in position
        filled_main_frame["dz"] = filled_main_frame.groupby("robot")["z"].diff()
        filled_main_frame["dy"] = filled_main_frame.groupby("robot")["y"].diff()
        filled_main_frame["dx"] = filled_main_frame.groupby("robot")["x"].diff()

        # Fill missing activity for 'drone_1' using EXPLORE events
        drone_1_indices = filled_main_frame[(filled_main_frame['robot'] == 'drone_1') & (pd.isna(filled_main_frame["activity"]))].index
        for i in range(filled_main_frame["run"].max()):
            explore_start_indices = filled_main_frame[(filled_main_frame['activity'] == 'EXPLORE') & (filled_main_frame["lifecycle"] == "START") & (filled_main_frame["run"] == i)].index
            explore_stop_indices = filled_main_frame[(filled_main_frame['activity'] == 'EXPLORE') & (filled_main_frame["lifecycle"] == "STOP") & (filled_main_frame["run"] == i)].index

            for j in range(len(explore_start_indices)):
                start_index = explore_start_indices[j]
                stop_index = explore_stop_indices[j]
                indices = drone_1_indices[(drone_1_indices >= start_index) & (drone_1_indices <= stop_index)]
                filled_main_frame.loc[indices, 'activity'] = 'EXPLORE'

        # Compute time since start per run
        for i in range(filled_main_frame["run"].max()):
            filled_main_frame.loc[filled_main_frame["run"] == i, "minutes_since_start"] = (
                (filled_main_frame.loc[filled_main_frame["run"] == i, "time"] - filled_main_frame.loc[filled_main_frame["run"] == i, "time"].min())
                .dt.total_seconds() / 60
            )

        # Apply custom function `has_payload`
        filled_main_frame["has_payload"] = filled_main_frame["payload"].apply(has_payload)

        # Propagate TAKEOFF activity
        for i in range(filled_main_frame["run"].max()):
            if i in [5, 23]:  # Skip problematic runs
                continue
            first_takeoff = filled_main_frame[(filled_main_frame["run"] == i) & (filled_main_frame["activity"] == "TAKEOFF")].index[0]
            first_explore = filled_main_frame[(filled_main_frame["run"] == i) & (filled_main_frame["activity"] == "EXPLORE")].index[0]

            # Fill missing drone_1 activity with TAKEOFF until first EXPLORE
            drone_1_indices = filled_main_frame[(filled_main_frame['robot'] == 'drone_1') & (pd.isna(filled_main_frame["activity"]))].index
            indices = drone_1_indices[(drone_1_indices >= first_takeoff) & (drone_1_indices <= first_explore)]
            filled_main_frame.loc[indices, 'activity'] = 'TAKEOFF'

        # Encode categorical variables
        filled_main_frame["robot"] = self.robot_encoder.transform(filled_main_frame[["robot"]])

        return filled_main_frame.fillna(0)  # Replace any remaining NaNs with 0
