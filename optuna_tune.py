import optuna
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='optuna_trials.log', level=logging.INFO)

# Load dataset
def load_dataset():
    data = pd.read_csv("processed_data.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Define Neural Network architecture
class IDSModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(IDSModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 2)  # Assuming binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Objective function for Optuna optimization
def objective(trial):
    # Load data
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    # Hyperparameter search space
    hidden_size1 = trial.suggest_int('hidden_size1', 32, 128)
    hidden_size2 = trial.suggest_int('hidden_size2', 32, 128)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    
    # Instantiate model
    model = IDSModel(input_size=X_train.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(test_labels, predicted)
    
    # Log trial details
    log_msg = (f"Trial {trial.number} - Accuracy: {accuracy:.4f} | "
               f"hidden_size1: {hidden_size1}, hidden_size2: {hidden_size2}, lr: {lr:.6f}")
    print(log_msg)
    logging.info(log_msg)
    
    return accuracy

# Main script
if __name__ == "__main__":
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)  # Set the number of trials
    
    print("Optimization completed.")
    print(f"Best trial: {study.best_trial.number} - Accuracy: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
