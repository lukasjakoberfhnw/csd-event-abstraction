from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from tabular_to_sequential import to_sequential

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
# sequence_length = 1  # Number of timesteps in each sequence
num_epochs = 5
learning_rate = 0.01

# Load preprocessing 4 dataset
preprocessing_file_4_path = os.path.join(os.path.dirname(__file__), '..', 'data', "tale-camerino", "from_massimiliano", "processed", "tale_data_preprocessed_4_train.csv")
df = pd.read_csv(preprocessing_file_4_path)
print(len(df))

df.dropna(inplace=True)

# get index of the end of the first run
first_run_index = df[df['run'] == 0].index[-1]
second_run_index = df[df['run'] == 1].index[-1]

# df = df.iloc[:20000]
df_train = df.iloc[0:first_run_index]
df_test = df.iloc[first_run_index:second_run_index]

print(len(df_train))
print(len(df_test))

X_train, y_train, labelEncoder_train = to_sequential(df_train, "sequential_data.pth", recreate=True)
X_test, y_test, labelEncoder_test = to_sequential(df_test, "sequential_data_test.pth", recreate=True)

# Check the shape of loaded tensors
print("Loaded Input Shape:", X_train.shape)  # (num_samples, sequence_length, num_features)
print("Loaded Target Shape:", y_train.shape)  # (num_samples, sequence_length)

# Instantiate Dataset
dataset = ActivityDataset(X_train, y_train)
batch_size = 32  # Adjust based on memory & dataset size
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_size = X_train.shape[2]

output_size = len(torch.unique(y_train))  # Number of classes
print("Unique classes:", torch.unique(y_train))  # Debugging step ---> there is a problem here...

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
# X_test, Y_test = torch.load("sequential_data_test.pth")

print("Testing the model...")
model.eval()  # Set model to evaluation mode
x_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

predicted = model(X_test)  # Get predictions
predicted_labels = torch.argmax(predicted, dim=2)  # Convert to class labels
print(f'Predicted activities: {predicted_labels.tolist()}')

# Unroll the sequences for evaluation
predicted_labels_unrolled = predicted_labels.view(-1)
y_test_unrolled = y_test.view(-1)
accuracy = (predicted_labels_unrolled == y_test_unrolled).float().mean()
print(f'Test Accuracy: {accuracy.item() * 100:.2f}%')

# Convert back to original labels
predicted_labels_unrolled = labelEncoder_test.inverse_transform(predicted_labels_unrolled)
y_test_unrolled = labelEncoder_test.inverse_transform(y_test_unrolled)

# Classification report
print(classification_report(y_test_unrolled, predicted_labels_unrolled))

