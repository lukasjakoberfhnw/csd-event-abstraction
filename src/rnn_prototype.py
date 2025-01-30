import pandas as pd
import os
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed")
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')

# LSTM model
class ActivityRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ActivityRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use last time step's output
        return out

# Dataset class
class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def rnn_train(df):
    # Encode labels
    label_encoder = LabelEncoder()
    df["activity"] = label_encoder.fit_transform(df["activity"])

    # Select features (All except "activity")
    features = df.drop(columns=["activity"]).astype(float)
    labels = df["activity"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActivityRNN(input_size=features.shape[1], hidden_size=64, num_layers=2, output_size=len(label_encoder.classes_)).to(device)

    # Define train/loader
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, y_train = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(Dataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(Dataset(X_test, y_test), batch_size=64, shuffle=False)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, predicted.cpu().numpy(), target_names=label_encoder.classes_))


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(os.path.join(data_path, 'tale_data_preprocessed_3_5_train.csv'))
    print(df.head())

    df = df.iloc[:1000]  # Limit the number of samples for faster training

    rnn_train(df)