import pandas as pd
import os
from sklearn.metrics import classification_report
import pycrfsuite


data_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed")
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')

# inspired by Massimilianos code ()

def prepare_sequences(data):
    sequences = []
    labels = []
    
    for _, group in data.groupby('run'):  # Group by `run` to create sequences
        xseq = group.drop(columns=['activity', 'run']).to_dict(orient='records')  # Feature sequence
        yseq = group['activity'].tolist()  # Label sequence

        if len(xseq) != len(yseq):  
            print(f"⚠️ Mismatch detected: len(features)={len(xseq)}, len(labels)={len(yseq)}")

        sequences.append(xseq)
        labels.append(yseq)
    
    return sequences, labels

def train_model():
    print("Training model")

    # Load the data
    data = pd.read_csv(os.path.join(data_path, 'tale_data_preprocessed_3_5_train.csv'))
    print("Data loaded")

    # Drop unpredictable activities
    data = data.loc[~data['activity'].isin(["LOW_BATTERY", "TIME_OUT", "RETURN_TO_BASE"])]
    print("Unpredictable activities dropped")

    # Split the data into training and test sets
    index_splitter = int(len(data) * 0.8)
    train_data = data.iloc[:index_splitter]
    test_data = data.iloc[index_splitter:]
    print("Data split into training and test sets")

    # Prepare sequences
    train_sequences, train_labels = prepare_sequences(train_data)

    # Instantiate the trainer and set its parameters
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params({
        'c1': 1,  # coefficient for L1 penalty
        'c2': 0.1,  # coefficient for L2 penalty
        'max_iterations': 500,  # Was 5000 before...
        'feature.possible_transitions': True
    })

    print("Trainer instantiated and parameters set")

    # Train the model
    for xseq, yseq in zip(train_sequences, train_labels):
        trainer.append(xseq, yseq)

    trainer.train('model.crfsuite')
    print("Model training completed")

def test_model():
    print("Validating model")
    
    crf_tagger = pycrfsuite.Tagger()
    crf_tagger.open('model.crfsuite')
    print("Model opened")

    # Load the test data
    data = pd.read_csv(os.path.join(data_path, 'tale_data_preprocessed_3_5_train.csv'))
    print("Data loaded")

    # Drop unpredictable activities
    data = data.loc[~data['activity'].isin(["LOW_BATTERY", "TIME_OUT", "RETURN_TO_BASE"])]
    print("Unpredictable activities dropped")

    # Split the data into training and test sets
    index_splitter = int(len(data) * 0.8)
    test_data = data.iloc[index_splitter:]
    print("Test data extracted")

    # Prepare test sequences
    test_sequences, test_labels = prepare_sequences(test_data)  # Reusing the function from train_model()

    print("Test sequences prepared")

    # Run CRF tagger on test sequences
    all_true, all_pred = [], []
    
    for xseq, yseq in zip(test_sequences, test_labels):
        predicted = crf_tagger.tag(xseq)  # Predict labels
        all_true.extend(yseq)  # Flatten true labels
        all_pred.extend(predicted)  # Flatten predictions

    print("Testing completed")
    
    # Print evaluation metrics
    print(classification_report(all_true, all_pred))

def main():
    train_model()
    test_model()

if __name__ == '__main__':
    main()