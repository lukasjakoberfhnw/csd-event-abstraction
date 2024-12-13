import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# load the data
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


    # make sure all columns are in the correct format
    # full_df['time'] = pd.to_datetime(full_df['time'])
    full_df['activity'] = full_df['activity'].astype('category')
    full_df['has_payload'] = full_df['has_payload'].astype('category')
    full_df['robot'] = full_df['robot'].astype('category')

    # one hot encode the activity, has_payload and robot columns
    full_df = pd.get_dummies(full_df, columns=['has_payload', 'robot'])

    # split full_df into train and test with the last 4 runs as test
    train_df = full_df[full_df['run'] < 30]
    test_df = full_df[full_df['run'] >= 30]

    # # remove columns that are not needed
    train_df = train_df.drop(columns=['run', 'payload', 'time', 'lifecycle', 'x', 'y', 'z'])
    print("Train columns", train_df.columns)
    train_X = train_df.drop(columns=['activity']).to_numpy()
    train_y = train_df['activity'].to_numpy()

    test_df = test_df.drop(columns=['run', 'payload', 'time', 'lifecycle', 'x', 'y', 'z'])
    print("Test columns", test_df.columns)
    test_X = test_df.drop(columns=['activity']).to_numpy()
    test_y = test_df['activity'].to_numpy()

    # # create a decision tree classifier
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier(n_estimators=50, max_depth=10)
    # clf = AdaBoostClassifier(n_estimators=50)
    # clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=50)

    # fit the classifier
    clf.fit(train_X, train_y)

    # predict the test data
    predictions = clf.predict(test_X)

    # calculate the accuracy
    # accuracy = (test_df['activity'] == predictions).sum() / len(test_df)
    # print(f"Accuracy: {accuracy}")

    # print sklearn metrics for the classifier
    print(classification_report(test_y, predictions))

if __name__ == '__main__':
    main()