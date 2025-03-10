# not active - just for testing purposes

import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load("./models/pipeline_activity_prediction.pkl")

# Load the activity encoder
activity_encoder = joblib.load("./models/activity_encoder.pkl")

data = pd.read_csv("./data/tale-camerino/from_massimiliano/processed/tale_data_raw_aggregated.csv")  # Load data

# Select a single row of data
features = data.iloc[0:1]

# Make predictions
encoded_prediction = pipeline.predict(features).tolist()

# Decode prediction back to original activity names
prediction = activity_encoder.inverse_transform(encoded_prediction).tolist()

print("prediction: ", prediction)