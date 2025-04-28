import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from typing import Callable, Iterable

class QuantileLoss:
    def __init__(self, q: Iterable):
        self.q = q

    def __call__(self, preds, target):
        errors = target - preds[:, 0]
        loss_0 = torch.max((self.q[0] - 1) * errors, self.q[0] * errors)
        errors = target - preds[:, 1]
        loss_1 = torch.max((self.q[1] - 1) * errors, self.q[1] * errors)
        return torch.mean(loss_0 + loss_1)

class DenseNetwork(nn.Module):
    def __init__(
            self, 
            n_layers: int, 
            hidden_width: int, 
            out_dim: int, 
            input_dim=180, 
            dropout=0.1
        ):

        super(DenseNetwork, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_width))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())

        # Add dropout and output layer
        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_width, out_dim)

    def forward(self, x):
        x = self.network(x)
        x = self.dropout(x)
        return self.out_layer(x)


class TrainingLoop():
    def __init__(
        self, 
        loss: Callable, 
        batch_size: int = 512, 
        n_epochs: int = 200,
        val_ppt: int = 0.1,
        lr: float = 0.001,
        patience_ppt: int = 0.1,

    ):
        self.loss = loss
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.val_ppt = val_ppt
        self.lr = lr
        self.patience = int(n_epochs * patience_ppt)

    def __call__(self, model: nn.Module, device: str, X: Iterable, y: Iterable):

        #pass data and model on gpu if available
        X = torch.Tensor(X).to(device) 
        y = torch.Tensor(y).to(device)
        
        #split
        X_train, X_val, y_train, y_val = split(X, y, test_size=self.val_ppt)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), 
            batch_size=self.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=self.batch_size
        )
        
        #set parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        best_loss = torch.inf
        patience_counter = 0

        # Training loop
        for epoch in range(self.n_epochs):
            for batch_X, batch_y in train_loader:
                model.train()
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_X, batch_y in val_loader:
                    val_outputs = model(batch_X)
                    val_loss += self.loss(val_outputs, batch_y).item()
                val_loss /= batch_y.shape[0]

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load the best model state
        model.load_state_dict(best_model_state)


class DenseClassifier(ClassifierMixin):
    def __init__(self, n_layers: int, hidden_width: int, 
        input_dim=180, dropout=0.1, nclasses=3):

        self.gpu = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DenseNetwork(n_layers, hidden_width, out_dim=nclasses, 
            input_dim=input_dim, dropout=dropout).to(self.gpu)
        self.loss = nn.CrossEntropyLoss()
        self.nclasses = 3
    
    def fit(self, X: Iterable, y: Iterable, **kwargs):
        enc = OneHotEncoder()
        enc.fit(np.arange(self.nclasses).reshape(-1, 1))
        target = enc.transform(y.reshape(-1, 1)).toarray()
        loop = TrainingLoop(self.loss, **kwargs)
        loop(self.model, self.gpu, X, target)

    def predict(self, X: Iterable) -> np.array:
        X = torch.tensor(X).to(self.gpu)
        self.model.eval()
        output = torch.argmax(self.model.to(self.gpu)(X.float()), axis=1)
        return output.detach().cpu().numpy()

    def predict_proba(self, X: Iterable) -> np.array:
        X = torch.tensor(X).to(self.gpu).float()
        self.model.eval()
        output = torch.softmax(self.model.to(self.gpu)(X), axis=1)
        return output.detach().cpu().numpy()


class DenseRegressor(RegressorMixin):
    def __init__(self, n_layers: int, hidden_width: int,
        quantile: Iterable, input_dim=180, dropout=0.1):

        self.gpu = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DenseNetwork(n_layers, hidden_width, out_dim=2, 
            input_dim=input_dim, dropout=dropout).to(self.gpu)
        self.loss = QuantileLoss(quantile)
    
    def fit(self, X: Iterable, y: Iterable, **kwargs):
        loop = TrainingLoop(self.loss, **kwargs)
        loop(self.model, self.gpu, X, y)

    def predict(self, X: Iterable) -> np.array:
        X = torch.tensor(X).to(self.gpu)
        return self.model.to(self.gpu)(X.float()).cpu().detach().numpy()


