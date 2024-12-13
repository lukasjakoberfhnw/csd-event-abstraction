import os
import pm4py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')
tale_XES_file_location = os.path.join(data_path, 'tale-camerino', 'MRS.xes')

def main():
    # load tale xes file
    log = pm4py.read_xes(tale_XES_file_location)
    log_df = pm4py.convert_to_dataframe(log)

    # preprocess payload to be a new class when nan
    log_df['payload'] = log_df['payload'].fillna('NoPayload')
    log_df['lifecycle:transition'] = log_df['lifecycle:transition'].fillna('NoTransition')
    log_df['lifecycle:state'] = log_df['lifecycle:state'].fillna('NoState')

    # preprocess time to be in seconds
    log_df['pd_time'] = pd.to_datetime(log_df['time:timestamp']).astype(int) / 10**9

    train_data = log_df.iloc[:70000].copy()
    test_data = log_df.iloc[70000:75000].copy()

    print("Train Data Shape: ", train_data.shape)
    print("Test Data Shape: ", test_data.shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_data['concept:name'])

    train_data['concept:name'] = label_encoder.transform(train_data['concept:name'])
    test_data['concept:name'] = label_encoder.transform(test_data['concept:name'])

    # transform robot column to categorical
    train_data['robot'] = pd.Categorical(train_data['robot'], categories=train_data['robot'].unique()).codes
    test_data['robot'] = pd.Categorical(test_data['robot'], categories=test_data['robot'].unique()).codes

    # could preprocess the payload column as well
    # print number of unique values in the payload column
    print("Unique Values in Payload Column: ", train_data['payload'].nunique())

    train_data['payload'] = pd.Categorical(train_data['payload'], categories=train_data['payload'].unique()).codes
    test_data['payload'] = pd.Categorical(test_data['payload'], categories=test_data['payload'].unique()).codes
    train_data['lifecycle:transition'] = pd.Categorical(train_data['lifecycle:transition'], categories=train_data['lifecycle:transition'].unique()).codes
    test_data['lifecycle:transition'] = pd.Categorical(test_data['lifecycle:transition'], categories=test_data['lifecycle:transition'].unique()).codes
    train_data['lifecycle:state'] = pd.Categorical(train_data['lifecycle:state'], categories=train_data['lifecycle:state'].unique()).codes
    test_data['lifecycle:state'] = pd.Categorical(test_data['lifecycle:state'], categories=test_data['lifecycle:state'].unique()).codes

    X_train = train_data[['x', 'y', 'z', 'robot', 'payload', 'pd_time', 'lifecycle:transition', 'lifecycle:state']] # without payload for now
    y_train = train_data['concept:name']

    X_test = test_data[['x', 'y', 'z', 'robot', 'payload', 'pd_time', 'lifecycle:transition', 'lifecycle:state']]
    y_test = test_data['concept:name']

    decision_tree = AdaBoostClassifier()
    decision_tree.fit(X_train, y_train)

    y_pred = decision_tree.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    # accuracy with decision tree using 100% of the data --> 0.7323517402060865
    # this with using only X, Y, Z, and Robot columns

    # accuracy using the payload column as well --> 0.9336583959839551
    # quite important column to use

    # adding time with all samples: 0.9441788965484111

    # using random forest classifier, all samples, all features: 0.9441788965484111
    # same as a normal tree... maybe impossible to get better based on the data
    # using limited samples (50000) and all features: 0.1418264895088078
    # makes sense because we try to predict the end of the process with the beginning of the process
    # from 50000 to 60000: accuracy increases to 0.2942
    # from 70000 to 75000: accuracy increases to 0.4878
    # with AdaBoost (70-75) -> accuracy: 0.732
    # adaboost with all samples: 0.739 - performs actually worse than the decision tree

    # normal trees with 70-75: Accuracy:  0.508

    # integrating the lifecycle feature -> idk if we are allowed to use them...
    # only increases accuracy to 0.960

    # using the two lifecycle features with adaboost and 70-75: still 0.732

if __name__ == '__main__':
    main()