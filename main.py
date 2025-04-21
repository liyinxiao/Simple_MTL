import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 16
shared_hidden = [64, 32]
task_hidden = [16, 8]
batch_size = 32
epochs = 10


# -------- Model Definition --------
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskModel, self).__init__()

        # Shared bottom
        layers = []
        last_dim = input_dim
        for h in shared_hidden:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.shared_bottom = nn.Sequential(*layers)

        # CTR tower
        ctr_layers = []
        for h in task_hidden:
            ctr_layers.append(nn.Linear(last_dim, h))
            ctr_layers.append(nn.ReLU())
            last_dim = h
        ctr_layers.append(nn.Linear(last_dim, 1))
        ctr_layers.append(nn.Sigmoid())
        self.ctr_tower = nn.Sequential(*ctr_layers)

        # CVR tower
        last_dim = shared_hidden[-1]  # Reset from shared output
        cvr_layers = []
        for h in task_hidden:
            cvr_layers.append(nn.Linear(last_dim, h))
            cvr_layers.append(nn.ReLU())
            last_dim = h
        cvr_layers.append(nn.Linear(last_dim, 1))
        cvr_layers.append(nn.Sigmoid())
        self.cvr_tower = nn.Sequential(*cvr_layers)

    def forward(self, x):
        shared = self.shared_bottom(x)
        ctr = self.ctr_tower(shared)
        cvr = self.cvr_tower(shared)
        return ctr, cvr


# -------- Dummy Data --------
num_samples = 1000
X = torch.randn(num_samples, input_dim)
y_ctr = torch.randint(0, 2, (num_samples, 1)).float()
y_cvr = torch.randint(0, 2, (num_samples, 1)).float()

dataset = TensorDataset(X, y_ctr, y_cvr)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------- Training --------
model = MultiTaskModel(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(epochs):
    for x_batch, y_ctr_batch, y_cvr_batch in loader:
        x_batch = x_batch.to(device)
        y_ctr_batch = y_ctr_batch.to(device)
        y_cvr_batch = y_cvr_batch.to(device)

        optimizer.zero_grad()
        ctr_pred, cvr_pred = model(x_batch)

        loss_ctr = criterion(ctr_pred, y_ctr_batch)
        loss_cvr = criterion(cvr_pred, y_cvr_batch)
        loss = loss_ctr + loss_cvr  # Simple sum of losses

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------- Inference --------
model.eval()
with torch.no_grad():
    input = torch.randn(1, input_dim).to(device)
    ctr_pred, cvr_pred = model(input)
    print(f"Predicted CTR: {ctr_pred.item():.4f}, Predicted CVR: {cvr_pred.item():.4f}")
