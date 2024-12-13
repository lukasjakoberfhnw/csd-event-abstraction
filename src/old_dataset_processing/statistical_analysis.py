import os
import pm4py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')
tale_XES_file_location = os.path.join(data_path, 'tale-camerino', 'MRS.xes')

def find_pattern(lst):
    for i in range(1, len(lst)):
        pattern = lst[:i]
        if pattern * (len(lst) // len(pattern)) == lst[:len(pattern) * (len(lst) // len(pattern))]:
            return pattern
    return None


def main():
    # load data
    log = pm4py.read_xes(tale_XES_file_location)
    log_df = pm4py.convert_to_dataframe(log)

    # split by robot
    drone = log_df[log_df['robot'] == 'drone']
    tractor1 = log_df[log_df['robot'] == 'tractor_1']
    tractor2 = log_df[log_df['robot'] == 'tractor_2']

    print(drone.shape)
    print(tractor1.shape)
    print(tractor2.shape)

    # check if the lifecycle:transition always follows the same pattern
    print(drone['lifecycle:transition'].unique())

    pattern = find_pattern(drone['lifecycle:transition'].tolist())
    print(pattern)

    # does not hold for drone --> there is no start -> nan -> complete

if __name__ == '__main__':
    main()