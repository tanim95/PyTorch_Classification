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


class MLPClassifier(nn.Module):  # Multilayer Perceptrons
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPClassifier, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        # making layers
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        # connecting them sequentinally
        self.model = nn.Sequential(*layers)
        # print(layers)
        # print(sizes)

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


# for showing "train_dataset"
# for i in range(3):
#     sample = train_dataset[i]
#     print(f"Sample {i + 1}:")
#     print("Features:", sample[0])
#     print("Label:", sample[1])
#     print()


# for showing "train_loader"
# for batch in train_loader:
#     inputs, labels = batch
#     print(inputs[:5,], labels[:5,])
#     print("Batch:")
#     print("Batch Size:", len(inputs))
#     print("Features Shape:", inputs.shape)
#     print("Labels Shape:", labels.shape)


# Instantiate the model
model = MLPClassifier(input_size=2, hidden_sizes=[
                      64, 32, 64, 32, 64], output_size=3)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.007)

# Training the Model
epochs = 300
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Validate the Model
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    _, val_predicted = torch.max(val_outputs, 1)

#  Validation Accuracy
val_accuracy = accuracy_score(y_val.numpy(), val_predicted.numpy())
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Test the Model
model.eval()
with torch.no_grad():  # Forward pass without tracking gradients
    test_outputs = model(X_test)
    _, test_predicted = torch.max(test_outputs, 1)

# Test Accuracy
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
    # plt.show()


# Plot Decision Boundaries on Validation Set
plot_decision_boundary(X_val.numpy(), y_val.numpy(), model,
                       title='Validation Set Decision Boundaries')

# Plot Decision Boundaries on Test Set
plot_decision_boundary(X_test.numpy(), y_test.numpy(),
                       model, title='Test Set Decision Boundaries')


# Save a trained model
torch.save(model.state_dict(), 'Classifier1.pth')


loaded_model = MLPClassifier(input_size=2, hidden_sizes=[
    64, 32, 64, 32, 64], output_size=3)
loaded_model.load_state_dict(torch.load('Classifier1.pth'))

# Set the model to evaluation mode
loaded_model.eval()
with torch.no_grad():
    prediction = torch.max(loaded_model(X_test), 1)

print(prediction)


# /...........................Regressor ..............

class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRegressor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size[0])
        self.layer2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.layer3 = nn.Linear(hidden_size[1], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x


# Generating some dummy data for demonstration purposes
# Replace this with your actual data
input_size = 10
hidden_size = hidden_sizes = [64, 32]
output_size = 1
num_samples = 1000

# random input and output tensors
inputs = torch.randn(num_samples, input_size)
true_outputs = torch.randn(num_samples, output_size)

# Define an MLPRegressor model
mlp_regression_model = MLPRegressor(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_regression_model.parameters(), lr=0.001)

# Create a DataLoader for the dataset
dataset = TensorDataset(inputs, true_outputs)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop (not the full training loop for brevity)
num_epochs = 10
for epoch in range(num_epochs):
    for i, (input_batch, output_batch) in enumerate(data_loader):
        optimizer.zero_grad()

        # Forward pass
        predicted_outputs = mlp_regression_model(input_batch)

        # Calculate loss
        loss = criterion(predicted_outputs, output_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# Save the trained MLPRegressor model
torch.save(mlp_regression_model.state_dict(), 'saved_mlp_regression_model.pth')

# Load the saved MLPRegressor model
loaded_mlp_regression_model = MLPRegressor(
    input_size, hidden_size, output_size)
loaded_mlp_regression_model.load_state_dict(
    torch.load('saved_mlp_regression_model.pth'))
loaded_mlp_regression_model.eval()

# Example input data for regression prediction
# New input tensor of shape (1, input_size)
new_input_data_regression = torch.randn(1, input_size)

# Set the model to evaluation mode
loaded_mlp_regression_model.eval()

# Perform regression prediction using the loaded MLP model
with torch.no_grad():
    predicted_output = loaded_mlp_regression_model(new_input_data_regression)

# The predicted_output tensor contains the model's regression prediction
# .item() retrieves the value from a tensor with a single element
print("Predicted Output:", predicted_output.item())


# ................Detailed training loop...............
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Convert data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# # Create DataLoader for training and validation
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# # Define your MLPRegressor model
# class MLPRegressor(nn.Module):
#     # Your model definition goes here, similar to the previous examples

# # Initialize the model, criterion, optimizer
# input_size = X_train.shape[1]  # Adjust this according to your input features
# hidden_size = 50  # Change as needed
# output_size = 1  # For regression, output size should typically be 1
# mlp_regression_model = MLPRegressor(input_size, hidden_size, output_size)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(mlp_regression_model.parameters(), lr=0.001)

# # Training parameters
# num_epochs = 50
# best_val_loss = float('inf')
# patience = 5
# early_stopping_counter = 0

# # Training loop with validation and early stopping
# for epoch in range(num_epochs):
#     # Training phase
#     mlp_regression_model.train()
#     train_losses = []
#     for i, (inputs, targets) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = mlp_regression_model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())

#     # Validation phase
#     mlp_regression_model.eval()
#     val_losses = []
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             outputs = mlp_regression_model(inputs)
#             val_loss = criterion(outputs, targets)
#             val_losses.append(val_loss.item())

#     # Calculate average losses
#     avg_train_loss = sum(train_losses) / len(train_losses)
#     avg_val_loss = sum(val_losses) / len(val_losses)

#     # Early stopping
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(mlp_regression_model.state_dict(), 'best_model.pth')
#         early_stopping_counter = 0
#     else:
#         early_stopping_counter += 1

#     # Print training/validation statistics
#     print(f"Epoch [{epoch+1}/{num_epochs}] - "
#           f"Avg. Train Loss: {avg_train_loss:.4f}, "
#           f"Avg. Val Loss: {avg_val_loss:.4f}")

#     # Early stopping condition
#     if early_stopping_counter >= patience:
#         print("Early stopping triggered!")
#         break

# # After the loop, load the best model
# loaded_model = MLPRegressor(input_size, hidden_size, output_size)
# loaded_model.load_state_dict(torch.load('best_model.pth'))
# loaded_model.eval()
