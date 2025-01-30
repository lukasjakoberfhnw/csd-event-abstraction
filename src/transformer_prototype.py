import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ✅ Load Dataset
data_path = "your_dataset.csv"  # Replace with actual dataset path
df = pd.read_csv(data_path)

# ✅ Encode Labels
label_encoder = LabelEncoder()
df["activity"] = label_encoder.fit_transform(df["activity"])

# ✅ Select Features (All except "activity")
features = df.drop(columns=["activity"]).astype(float)
labels = df["activity"].values

# ✅ Normalize Data
scaler = StandardScaler()
X = scaler.fit_transform(features)

# ✅ Split Data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# ✅ Convert to Torch Tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

# ✅ PyTorch Dataset
class ActivityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ActivityDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(ActivityDataset(X_test, y_test), batch_size=64, shuffle=False)

# ✅ Transformer Model for Tabular Data
class TabularTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TabularTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Map tabular features to Transformer input size
        self.pos_encoder = nn.Parameter(torch.randn(1, input_dim, d_model))  # Positional encoding

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert tabular data to dense representation
        x = x + self.pos_encoder  # Add positional encoding
        x = self.transformer_encoder(x)  # Transformer encoder
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)  # Classification layer

# ✅ Model Setup
input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TabularTransformer(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# ✅ Training Loop
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

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# ✅ Evaluation
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
