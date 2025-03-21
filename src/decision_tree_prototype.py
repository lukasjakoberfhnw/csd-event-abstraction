import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# load the data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed")
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')

dtype = {'time': 'str', 'activity': 'category', 'x': 'float', 'y': 'float', 'z': 'float', 'robot': 'category', 'run': 'int', 'has_payload': 'category'}


def main():
    chosen_preprocessing_method = 4
    preprocessing_methods = ['1_train.csv', '2_train.csv', '3_train.csv', '3_5_train.csv', '4_train.csv', '4_train_without_unpredictable.csv']
    for i, method in enumerate(preprocessing_methods):
        print(f"{i}: {method}")
        # if(i < 2):
        #     continue
        preprocessing_file = 'tale_data_preprocessed_' + method

        full_df = pd.read_csv(os.path.join(data_path, preprocessing_file))

        # drop unpredictable activities: ["LOW_BATTERY", "TIME_OUT", "RETURN_TO_BASE"]
        full_df = full_df.loc[~full_df['activity'].isin(["LOW_BATTERY", "TIME_OUT", "RETURN_TO_BASE"])]

        # print("Loaded full dataset")

        index_splitter = int(len(full_df) * 0.8)
        train_df = full_df.iloc[:index_splitter]
        test_df = full_df.iloc[index_splitter:]

        # print("Train columns", train_df.columns)

        # drop nan values - only for adaboost 
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        train_X = train_df.drop(columns=['activity']).to_numpy()
        train_y = train_df['activity'].to_numpy()

        # test_df = test_df.drop(columns=test_remover)
        # print("Test columns", test_df.columns)
        test_X = test_df.drop(columns=['activity']).to_numpy()
        test_y = test_df['activity'].to_numpy()

        # # create a decision tree classifier
        # clf = DecisionTreeClassifier(max_depth=10)
        # clf = RandomForestClassifier(n_estimators=50, max_depth=10)
        # clf = AdaBoostClassifier(n_estimators=50)
        clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=50)

        # fit the classifier
        clf.fit(train_X, train_y)

        # store clf in output folder
        joblib.dump(clf, os.path.join(output_path, 'decision_tree_classifier.joblib'))

        # predict the test data
        predictions = clf.predict(test_X)

        # calculate the accuracy
        # accuracy = (test_df['activity'] == predictions).sum() / len(test_df)
        # print(f"Accuracy: {accuracy}")

        # print sklearn metrics for the classifier
        print(classification_report(test_y, predictions))

        # draw the decision tree using matplotlib --- only if decision tree
        # from sklearn.tree import plot_tree
        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots(figsize=(20, 10))
        # plot_tree(clf, ax=ax, feature_names=train_df.drop(columns=['activity']).columns, class_names=clf.classes_, filled=True)
        # plt.show()

        # # store the rules of the decision tree in a text file
        # from sklearn.tree import export_text
        # rules = export_text(clf, feature_names=train_df.drop(columns=['activity']).columns.tolist())
        
        # # use random codes instead of class in the rules
        # classes = clf.classes_
        # for i, c in enumerate(classes):
        #     rules = rules.replace(str(c), f"class_{i}")

        # print(classes)

        # with open(os.path.join(output_path, 'decision_tree_rules.txt'), 'w') as f:
        #     f.write(rules)


if __name__ == '__main__':
    main()