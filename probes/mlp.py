import torch
from torch import nn
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import TensorDataset, DataLoader
import warnings


class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self, input_size, hidden_size=16, output_size=1,
            learning_rate=1e-3, epochs=20, patience=5, weight_decay=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = MLP(input_size, hidden_size, output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(self.model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)

        self.validation_scores = []

    def fit(self, X, y, batch_size=512):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42)

        y_train = torch.tensor(y_train).float()
        if self.output_size == 1:
            y_train = y_train.reshape(-1, 1)

        train_data = TensorDataset(X_train.clone().detach().float(), y_train)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)

        X_val = X_val.clone().detach().float().to(self.device)
        y_val = torch.tensor(y_val).float().to(self.device)

        if self.output_size == 1:
            y_val = y_val.reshape(-1, 1)

        early_stop_count = 0
        min_val_loss = float('inf')

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

            # Validation loss for early stopping
            val_outputs = self.model(X_val)
            val_loss = self.criterion(val_outputs, y_val)
            self.validation_scores.append(val_loss.item())

            if val_loss.item() < min_val_loss:
                min_val_loss = val_loss.item()
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.patience:
                print(f'Early stopping on epoch {epoch}')
                break

        return self

    def predict(self, X):
        X = X.float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
        outputs = outputs.cpu().numpy()
        if self.output_size == 1:
            outputs = outputs.flatten()
        return outputs


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
