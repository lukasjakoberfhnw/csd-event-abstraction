# don't use this file, it's just for testing purposes

import os
import joblib
import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from preprocessing.preprocessor_4 import ManualPreprocessingTransformer

# load the preprocessing function from the preprocessing folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocessing')))
from preprocessing.pipelines import preprocess_manual_preparation

# Define paths
TALE_PROCESSED_PATH = "C:/Users/lukas/dev/camerino/csd/event_abstraction_csd/data/tale-camerino/from_massimiliano/processed"
MODEL_PATH = "C:/Users/lukas/dev/camerino/csd/event_abstraction_csd/models"

# Ensure the model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Load dataset
df = pd.read_csv(os.path.join(TALE_PROCESSED_PATH, "tale_data_raw_aggregated.csv"))

# Encode the target variable (activity)
activity_encoder = LabelEncoder()
df["activity"] = activity_encoder.fit_transform(df["activity"])

# Save the activity encoder for decoding predictions later
activity_encoder_path = os.path.join(MODEL_PATH, "activity_encoder.pkl")
joblib.dump(activity_encoder, activity_encoder_path)
print(f"Activity encoder saved to: {activity_encoder_path}")

print(df.head())

# Separate features (X) and target (y)
X = df # all columns
y = df["activity"]  # Encoded target

# Create a pipeline with preprocessing + model - for training
pipeline_train = Pipeline([
    ("manual_preprocessing", ManualPreprocessingTransformer(preprocessing_function=preprocess_manual_preparation, train=False)),  # Custom feature engineering
    ("model", DecisionTreeClassifier(max_depth=10))  # ML model
])

# Train the pipeline
pipeline_train.fit(X, y)


# print pipeline steps
print(pipeline_train.steps)

# print pipeline results
print(pipeline_train)


# Save the entire pipeline
pipeline_path = os.path.join(MODEL_PATH, "pipeline_activity_prediction.pkl")
joblib.dump(pipeline_train, pipeline_path)
print(f"Pipeline saved to: {pipeline_path}")
