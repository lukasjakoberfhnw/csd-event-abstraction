import os
import pm4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from matplotlib.animation import FuncAnimation, FFMpegWriter

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')
tale_XES_file_location = os.path.join(data_path, 'tale-camerino', 'MRS.xes')

if __name__ == '__main__':
    # read xes file
    log = pm4py.read_xes(tale_XES_file_location)
    log_df = pm4py.convert_to_dataframe(log)

    print("Event Log Size: ", log_df.shape)

    print("Event Log Head:")
    print(log_df.head())

    print("Event Log Info:")
    print(log_df.info())

    print("Event Log Describe:")
    print(log_df.describe())

    # get unique values of the comumns with type object
    for column in log_df.select_dtypes(include=[object]).columns:
        print(column, log_df[column].unique())


    # possible approaches
    # 1. supervised learning for predicting the high-level activity
    # 2. unsupervised learning for clustering the activities
    # 3. custom method for finding the critical points in the dataset -> using graphs and networkx 
    # the most frequent path

    # most likely makes sense to find the points where unexpected change happens --> change is unpredictable from the previous data
    processing_df = log_df[['time:timestamp', 'x', 'y', 'z', 'robot', 'concept:name']].copy()
    processing_df['pd_time'] = pd.to_datetime(processing_df['time:timestamp'])
    processing_df['time_in_seconds'] = pd.to_datetime(processing_df['pd_time']).astype(int)/ 10**9
    processing_df['robot'] = pd.Categorical(processing_df['robot'], categories=processing_df['robot'].unique()).codes

    # print(processing_df['robot'].iloc[0:50])

    # columns = Index(['time:timestamp', 'concept:name', 'lifecycle:transition', 'payload',
    #    'x', 'y', 'z', 'lifecycle:state', 'robot', 'event_id', 'msg:id',
    #    'case:concept:name'],
    #   dtype='object')

    numpy_data = processing_df[['x', 'y', 'z', 'robot']].to_numpy()

    clustering = DBSCAN(eps=0.7, min_samples=3).fit(numpy_data)

    # clustering = KMeans(n_clusters=15).fit(numpy_data) 

    # too memory intensive
    # clustering = AgglomerativeClustering(n_clusters=15).fit(numpy_data)

    # print unique number of labels
    print(clustering.labels_)
    print(np.unique(clustering.labels_))

    # plot x, y, and z to see the distribution of the data in a 3d scatter figure 
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(processing_df['x'], processing_df['y'], processing_df['z'], c=clustering.labels_, cmap='viridis')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    # difference to actual activity shown in concept:name
    # add row to the dataframe with the cluster label
    processing_df['cluster'] = clustering.labels_

    # cross check labels with the actual activity if concept:name == cluster label
    processing_df.to_csv(os.path.join(output_path, 'dbscan2.csv'), index=False)

    # log_df.to_csv(os.path.join(output_path, 'log_df.csv'), index=False)

    drone_data = processing_df[processing_df['robot'] == 0].iloc[0:300]
    tractor1_data = processing_df[processing_df['robot'] == 1].iloc[0:300]
    tractor2_data = processing_df[processing_df['robot'] == 2].iloc[0:300]

    # plot the movement of the drone in a 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(min(drone_data['x']), max(drone_data['x']))
    ax.set_ylim(min(drone_data['y']), max(drone_data['y']))
    ax.set_zlim(min(drone_data['z']), max(drone_data['z']))

    drone_scatter = ax.scatter([], [], [], c='blue', s=50, label='Drone')
    tractor1_scatter = ax.scatter([], [], [], c='red', s=50, label='Tractor 1')
    tractor2_scatter = ax.scatter([], [], [], c='green', s=50, label='Tractor 2')

    # Update function for animation
    def update(frame):
        # Update drone data
        current_drone_data = drone_data.iloc[:frame+1]
        drone_scatter._offsets3d = (current_drone_data['x'], current_drone_data['y'], current_drone_data['z'])
        
        # Update tractor1 data
        current_tractor1_data = tractor1_data.iloc[:frame+1]
        tractor1_scatter._offsets3d = (current_tractor1_data['x'], current_tractor1_data['y'], current_tractor1_data['z'])
        
        # Update tractor2 data
        current_tractor2_data = tractor2_data.iloc[:frame+1]
        tractor2_scatter._offsets3d = (current_tractor2_data['x'], current_tractor2_data['y'], current_tractor2_data['z'])
        
        return drone_scatter, tractor1_scatter, tractor2_scatter
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(drone_data), blit=False)

    # Save or display
    writer = FFMpegWriter(fps=10, metadata={'artist': 'Lukas Jakober'}, bitrate=1800)
    ani.save('3d_scatter_animation.mp4', writer=writer)  # Save as video
    plt.show()  # Display interactively

    # probably not correct -> timestamps might be wrong