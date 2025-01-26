from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib
import pandas as pd
from sklearn.metrics import classification_report
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocessing')))
from preprocessing.pipelines import preprocess_manual_preparation


def check_prediction_using_stored_clf():
    TALE_PROCESSED_PATH = "./data/tale-camerino/from_massimiliano/processed"
    data = pd.read_csv(f"{TALE_PROCESSED_PATH}/tale_data_raw_aggregated.csv")

    # take the last run as test data
    test_data = data[data['run'] == data['run'].max()]
    # test_X = test_data.drop(columns=['activity', 'run'])
    test_data_y = test_data['activity'].fillna("IDLE")

    # preprocess the test data
    test_X = preprocess_manual_preparation(test_data, train=False).to_numpy()

    # load the classifier
    clf = joblib.load("./output/decision_tree_classifier.joblib")

    # predict the test data
    predictions = clf.predict(test_X)

    print(len(predictions))
    print(len(test_data['activity']))
    print(predictions)

    # add predictions to the test data
    test_data['predicted_activity'] = predictions
    test_data["actual_activity"] = test_data_y
    test_data.to_csv(f"{TALE_PROCESSED_PATH}/tale_data_raw_aggregated_with_predictions.csv")

    # compare with filled data
    filled_data = preprocess_manual_preparation(test_data, train=True)
    filled_data["predicted_activity"] = predictions
    filled_data["actual_activity"] = test_data_y

    filled_data.to_csv(f"{TALE_PROCESSED_PATH}/tale_data_preprocessed_4_train_with_predictions_postprocessing.csv")

    # compare the predictions with the actual values
    # print("Classification report:")
    # print(classification_report(test_data['activity'], predictions))

if __name__ == "__main__":
    check_prediction_using_stored_clf()