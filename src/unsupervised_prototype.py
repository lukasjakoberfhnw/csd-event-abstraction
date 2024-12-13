import pandas as pd
import os
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.cluster import completeness_score

# load the data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed")
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')

dtype = {'time': 'str', 'activity': 'category', 'x': 'float', 'y': 'float', 'z': 'float', 'robot': 'category', 'run': 'int', 'has_payload': 'category'}

def main():
    full_df = pd.read_csv(os.path.join(data_path, "full_dataset.csv"), dtype=dtype)

    # full_df['time'] = pd.to_datetime(full_df['time'])
    full_df['activity'] = full_df['activity'].astype('category')
    full_df['has_payload'] = full_df['has_payload'].astype('category')
    full_df['robot'] = full_df['robot'].astype('category')

    # one hot encode the activity, has_payload and robot columns
    full_df = pd.get_dummies(full_df, columns=['has_payload', 'robot'])

    # split full_df into train and test with the last 4 runs as test
    train_df = full_df[full_df['run'] < 30]
    test_df = full_df[full_df['run'] >= 30]

    # smaller dataset for testing on laptop
    # train_df = full_df[full_df['run'] < 5]
    # test_df = full_df[(full_df['run'] >= 5) & (full_df['run'] < 7)]

    # # remove columns that are not needed
    train_df = train_df.drop(columns=['run', 'payload', 'time', 'lifecycle']) # , 'x', 'y', 'z'
    train_df = train_df.dropna() # there are some dx, dy, dz columns that are NaN, maybe the start of the run
    print("Train columns", train_df.columns)
    train_X = train_df.drop(columns=['activity']).to_numpy()
    train_y = train_df['activity'].to_numpy()

    test_df = test_df.drop(columns=['run', 'payload', 'time', 'lifecycle']) # , 'x', 'y', 'z'
    test_df = test_df.dropna()
    print("Test columns", test_df.columns)
    test_X = test_df.drop(columns=['activity']).to_numpy()
    test_y = test_df['activity'].to_numpy()

    # use DBSCAN to cluster the data
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # dbscan.fit(train_X)

    # use KMeans to cluster the data
    k_means_clust = KMeans(n_clusters=13)
    k_means_clust.fit(train_X)

    # predict the test data
    predictions = k_means_clust.fit_predict(test_X)
    
    # print the completeness score
    print(completeness_score(test_y, predictions))

if __name__ == '__main__':
    main()