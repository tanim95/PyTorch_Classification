import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

torch.manual_seed(42)

# Spiral Data


def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='int')
    for n in range(classes):
        ix = range(points * n, points * (n + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(n * 4, (n + 1) * 4,
                        points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = n
    return X, y

# Neural Network Model


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPClassifier, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        # making layers
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        # connectin them sequentinally
        self.model = nn.Sequential(*layers)
        print(layers)
        print(sizes)

    def forward(self, x):
        return self.model(x)


# Convert to PyTorch Tensors
X, y = spiral_data(100, 3)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Create DataLoader for efficient batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate the model
model = MLPClassifier(input_size=2, hidden_sizes=[64, 32], output_size=3)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the Model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Validate the Model
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    _, val_predicted = torch.max(val_outputs, 1)

# Calculate Validation Accuracy
val_accuracy = accuracy_score(y_val.numpy(), val_predicted.numpy())
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Test the Model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_predicted = torch.max(test_outputs, 1)

# Calculate Test Accuracy
test_accuracy = accuracy_score(y_test.numpy(), test_predicted.numpy())
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Visualize the Decision Boundaries


def plot_decision_boundary(X, y, model, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).argmax(dim=1)
    Z = Z.numpy().reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y,
                cmap=plt.cm.Spectral, edgecolors='k', s=40)
    plt.title(title)
    plt.show()


# Plot Decision Boundaries on Validation Set
plot_decision_boundary(X_val.numpy(), y_val.numpy(), model,
                       title='Validation Set Decision Boundaries')

# Plot Decision Boundaries on Test Set
plot_decision_boundary(X_test.numpy(), y_test.numpy(),
                       model, title='Test Set Decision Boundaries')
