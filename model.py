import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Define the Neural Network architecture
class IDSModel(nn.Module):
    def __init__(self, input_size):
        super(IDSModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Assuming binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load processed data
def load_processed_data(filepath):
    return pd.read_csv(filepath)

# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
def evaluate_model(model, train_loader, test_loader):
    model.eval()  # Set the model to evaluation mode
    
    # Evaluate on training data
    train_predictions, train_labels = [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.numpy())
            train_labels.extend(labels.numpy())
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    
    # Evaluate on test data
    test_predictions, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions.extend(predicted.numpy())
            test_labels.extend(labels.numpy())
    
    test_accuracy = accuracy_score(test_labels, test_predictions)
    
    return train_accuracy, test_accuracy

# Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load the model
def load_model(model_path, input_size):
    model = IDSModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

# Main code
if __name__ == "__main__":
    # Load dataset
    data = load_processed_data("processed_data.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(list(zip(test_data, test_labels)), batch_size=32, shuffle=False)

    # Instantiate the model, define loss and optimizer
    model = IDSModel(input_size=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, epochs=10)

    # Evaluate the model
    print("Evaluating the model...")
    train_accuracy, test_accuracy = evaluate_model(model, train_loader, test_loader)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save the model after training
    save_model(model, "ids_model.pth")

    # Optionally, load the saved model and evaluate again
    print("\nLoading the model again for inference...")
    loaded_model = load_model("ids_model.pth", input_size=X_train.shape[1])
    train_accuracy, test_accuracy = evaluate_model(loaded_model, train_loader, test_loader)
    print(f"Loaded Model Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Loaded Model Test Accuracy: {test_accuracy * 100:.2f}%")

    # Optionally, visualize the training and test accuracies over epochs
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(10):  # Run for the number of epochs
        train_accuracy, _ = evaluate_model(model, train_loader, train_loader)
        _, test_accuracy = evaluate_model(model, train_loader, test_loader)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
    # Plotting the accuracies
    plt.plot(range(10), train_accuracies, label='Train Accuracy')
    plt.plot(range(10), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
