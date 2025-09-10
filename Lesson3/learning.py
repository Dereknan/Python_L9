import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 1. Dataset

X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. Model

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 3. Training function

def train_model(optimiser_choice="SGD", lr=0.1, epochs=100):
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()

    # Choose optimiser
    if optimiser_choice == "SGD":
        optimiser = optim.SGD(model.parameters(), lr=lr)
    elif optimiser_choice == "Momentum":
        optimiser = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # momentum added
    elif optimiser_choice == "Adam":
        optimiser = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    losses = []
    for epoch in range(epochs):
        optimiser.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return losses

# 4. Compare optimisers

losses_sgd = train_model("SGD", lr=0.1, epochs=100)
losses_momentum = train_model("Momentum", lr=0.1, epochs=100)
losses_adam = train_model("Adam", lr=0.1, epochs=100)

# 5. Plot results

plt.plot(losses_sgd, label="SGD")
plt.plot(losses_momentum, label="SGD + Momentum")
plt.plot(losses_adam, label="Adam")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimiser Comparison")
plt.legend()
plt.show()
