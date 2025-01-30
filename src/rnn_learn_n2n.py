from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the many-to-many RNN model
class ManyToManyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManyToManyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # Many-to-many RNN
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer

    def forward(self, x):
        out, _ = self.rnn(x)  # RNN forward pass
        out = self.fc(out)  # Apply FC layer at each time step
        return out  # Shape: (batch_size, seq_len, output_size)

# Create a custom Dataset
class ActivityDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]  # Returns one sequence and its label

# Hyperparameters
# input_size = 1  # Number of features per timestep (e.g., sensor readings)
hidden_size = 10  # Number of hidden units
#output_size = 3  # Number of activity classes (e.g., "walking", "running", "standing")
sequence_length = 1  # Number of timesteps in each sequence
num_epochs = 10
learning_rate = 0.01

# Generate a toy dataset: Random runs as input, activities as output labels
num_samples = 50
# x_train = torch.rand(num_samples, sequence_length, input_size)  # Random runs (simulated)
# y_train = torch.randint(0, output_size, (num_samples, sequence_length))  # Random activity labels

# # Convert labels to one-hot encoding for multi-class classification
# y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=output_size).float()

# read data from preprocessing file
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the saved sequential dataset
X_tensor, Y_tensor = torch.load("sequential_data.pth")

# Check the shape of loaded tensors
print("Loaded Input Shape:", X_tensor.shape)  # (num_samples, sequence_length, num_features)
print("Loaded Target Shape:", Y_tensor.shape)  # (num_samples, sequence_length)

# Instantiate Dataset
dataset = ActivityDataset(X_tensor, Y_tensor)
batch_size = 32  # Adjust based on memory & dataset size
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_size = X_tensor.shape[2]

output_size = len(torch.unique(Y_tensor))  # Number of classes
print("Unique classes:", torch.unique(Y_tensor))  # Debugging step

# Model, loss, and optimizer
model = ManyToManyRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Multi-class classification loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for X_batch, Y_batch in train_loader:
        if torch.isnan(X_batch).any() or torch.isnan(Y_batch).any():
            print("Found NaN in batch, skipping...")
            continue

        optimizer.zero_grad()
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs.view(-1, output_size), Y_batch.view(-1))  # Flatten for loss calculation
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')


# Load test sequential data
X_test, Y_test = torch.load("sequential_data_test.pth")

x_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(Y_test, dtype=torch.long)

predicted = model(X_test)  # Get predictions
predicted_labels = torch.argmax(predicted, dim=2)  # Convert to class labels
print(f'Predicted activities: {predicted_labels.tolist()}')

# Unroll the sequences for evaluation
predicted_labels_unrolled = predicted_labels.view(-1)
y_test_unrolled = y_test.view(-1)
accuracy = (predicted_labels_unrolled == y_test_unrolled).float().mean()
print(f'Test Accuracy: {accuracy.item() * 100:.2f}%')

