import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # Simple RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for output

    def forward(self, x):
        out, hidden = self.rnn(x)  # RNN forward pass
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out

# Hyperparameters
input_size = 1  # One feature per time step
hidden_size = 10
output_size = 1  # Regression output
sequence_length = 5  # Number of time steps in each input sequence
num_epochs = 200
learning_rate = 0.01

# Generate a simple toy dataset (sin wave prediction)
data = torch.linspace(0, 10, steps=100)
x_train = []
y_train = []

# Ensure proper reshaping for sequence learning
for i in range(len(data) - sequence_length):
    x_train.append(data[i:i+sequence_length].view(sequence_length, 1))  # (seq_len, 1)
    y_train.append(data[i+sequence_length])  # Predict next value

x_train = torch.stack(x_train)  # Shape: (batch, seq_len, input_size)
y_train = torch.tensor(y_train).view(-1, 1)  # Shape: (batch, output_size)

# Model, loss, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the trained model
test_input = torch.linspace(10, 10 + sequence_length, steps=sequence_length).view(1, sequence_length, 1)  # New input sequence
predicted = model(test_input).item()
print(f'Predicted next value: {predicted:.4f}')

# Test the trained model with longer sequence that should predict a list of values
test_input = torch.linspace(10, 10 + sequence_length, steps=sequence_length).view(1, sequence_length, 1)  # New input sequence
predicted = model(test_input).tolist()
print(f'Predicted next values: {predicted}')