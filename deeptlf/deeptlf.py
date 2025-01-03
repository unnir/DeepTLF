import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

import xgboost as xgb
from tqdm import tqdm

from .tde import TreeDrivenEncoder


class DeepTFL(BaseEstimator):
    """
    A deep learning model based on XGBoost and a custom neural network.

    Parameters
    ----------
    n_est : int, optional
        Number of estimators for XGBoost model, default is 23.
    max_depth : int, optional
        Max depth for each tree in XGBoost, default is 4.
    n_epoch : int, optional
        Number of epochs for neural network training, default is 100.
    hidden_dim : int, optional
        Hidden layer dimensions for neural network, default is 128.
    drop : float, optional
        Dropout rate for neural network, default is 0.23.
    xgb_lr : float, optional
        Learning rate for XGBoost model, default is 0.5.
    n_layers : int, optional
        Number of layers in the neural network, default is 4.
    checkpoint_name : str, optional
        File name to save the neural network model, default is 'checkpoint.pt'.
    batch_size : int, optional
        Batch size for neural network training, default is 320.
    task : str, optional
        Type of machine learning task ('class' for classification, other values for regression), default is 'class'.
    debug : bool, optional
        Whether to print debugging information, default is False.

    Attributes
    ----------
    xgb_model : XGBClassifier or XGBRegressor
        Fitted XGBoost model.
    nn_model : NeuralNet
        Fitted neural network model.
    TDE_encoder : TreeDrivenEncoder
        Fitted Tree-Driven Encoder.
    input_shape : int
        Shape of the input feature space.
    device : torch.device
        Device used for computations ('cuda' or 'cpu').
    """

    def __init__(
        self,
        n_est=23,
        max_depth=4,
        drop=0.23,
        xgb_lr=0.5,
        batch_size=320,
        n_epoch=100,
        hidden_dim=256,
        n_layers=4,
        task="class",
        debug=False,
        checkpoint_name="checkpoint.pt",
    ):
        # Validate input parameters
        if not isinstance(n_est, int) or n_est <= 0:
            raise ValueError("n_est must be a positive integer")
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        if not isinstance(drop, (int, float)) or not 0 <= drop <= 1:
            raise ValueError("drop must be a float between 0 and 1")
        if not isinstance(xgb_lr, (int, float)) or xgb_lr <= 0:
            raise ValueError("xgb_lr must be a positive float")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(n_epoch, int) or n_epoch <= 0:
            raise ValueError("n_epoch must be a positive integer")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError("n_layers must be a positive integer")
        if task not in ["class", "reg"]:
            raise ValueError("task must be either 'class' or 'reg'")
        if not isinstance(debug, bool):
            raise ValueError("debug must be a boolean")
        if not isinstance(checkpoint_name, str) or not checkpoint_name:
            raise ValueError("checkpoint_name must be a non-empty string")

        self.n_est = n_est
        self.max_depth = max_depth
        self.n_epoch = n_epoch
        self.hidden_dim = hidden_dim
        self.drop = drop
        self.debug = debug
        self.xgb_lr = xgb_lr
        self.n_layers = n_layers
        self.checkpoint_name = checkpoint_name
        self.batch_size = batch_size
        self.task = task
        self.xgb_model = None
        self.nn_model = None
        self.TDE_encoder = TreeDrivenEncoder()
        self.input_shape = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Validate inputs
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Empty input data")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            raise ValueError("Input contains NaN values")
        if X_val is not None and y_val is not None:
            if len(X_val) != len(y_val):
                raise ValueError("X_val and y_val must have the same length")
            if np.any(np.isnan(X_val)) or np.any(np.isnan(y_val)):
                raise ValueError("Validation data contains NaN values")

        # Store target values for classification tasks
        if self.task == "class":
            self.last_y = y_train

        self.fit_xgb(X_train)
        trees = self.xgb_model.get_booster().get_dump(with_stats=False)
        self.TDE_encoder.fit(trees)
        enc_X_train = self.TDE_encoder.transform(X_train)
        self.input_shape = enc_X_train.shape[1]
        self.fit_nn(enc_X_train, y_train, X_val, y_val)

    def fit_xgb(self, X_train):
        # Using XGBRegressor for self-supervised learning.
        self.xgb_model = xgb.XGBRegressor(
            learning_rate=self.xgb_lr,
            n_jobs=-1,
            max_depth=self.max_depth,
            n_estimators=self.n_est,
        )
        # Using X_train as target for self-supervised learning.
        self.xgb_model.fit(X_train, X_train)

    def fit_nn(self, enc_X_train, y_train, X_val=None, y_val=None):
        if X_val is not None:
            enc_X_val = self.TDE_encoder.transform(X_val)
        else:
            enc_X_train, enc_X_val, y_train, y_val = train_test_split(
                enc_X_train, y_train, test_size=0.2, random_state=42
            )

        train_loader = DataLoader(
            dataset=myDataset(enc_X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=myDataset(enc_X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss() if self.task == "class" else nn.MSELoss()
        num_of_outputs = len(set(y_train)) if self.task == "class" else 1
        self.nn_model = NeuralNet(
            self.input_shape, self.hidden_dim, self.n_layers, num_of_outputs, self.drop
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.nn_model.parameters(), lr=1e-3)
        early_stopping = EarlyStopping(
            patience=20, verbose=self.debug, path=self.checkpoint_name
        )

        for epoch in tqdm(range(self.n_epoch), desc="Epochs"):
            self.nn_model.train()
            for batch_X, batch_y in train_loader:
                if self.task == "class":
                    batch_X, batch_y = batch_X.float().to(
                        self.device
                    ), batch_y.long().to(self.device)
                else:
                    batch_X, batch_y = batch_X.float().to(
                        self.device
                    ), batch_y.float().to(self.device)
                outputs = self.nn_model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if X_val is not None:
            self.nn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if self.task == "class":
                        batch_X, batch_y = batch_X.float().to(
                            self.device
                        ), batch_y.long().to(self.device)
                    else:
                        batch_X, batch_y = batch_X.float().to(
                            self.device
                        ), batch_y.float().to(self.device)
                    outputs = self.nn_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                val_loss /= len(val_loader)  # Average validation loss
                early_stopping(val_loss, self.nn_model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break  # Break out of the epoch loop

    def predict(self, X):
        # Validate input
        if len(X) == 0:
            raise ValueError("Empty input data")
        if np.any(np.isnan(X)):
            raise ValueError("Input contains NaN values")
        
        # Check if model exists and is initialized
        if self.nn_model is None:
            try:
                self.nn_model = NeuralNet(
                    self.input_shape,
                    self.hidden_dim,
                    self.n_layers,
                    1 if self.task == "reg" else len(np.unique(self.last_y)),
                    self.drop
                ).to(self.device)
            except AttributeError:
                raise RuntimeError("Model not fitted. Call fit() before predict()")
        
        try:
            self.nn_model.load_state_dict(torch.load(self.checkpoint_name))
        except FileNotFoundError:
            raise RuntimeError(f"Model checkpoint not found at {self.checkpoint_name}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

        self.nn_model.eval()  # Set model to evaluation mode
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size  # Ceiling division
        y_hats = []

        with torch.no_grad():  # Disable gradient computation for prediction
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_samples)

                batch_X = X[start_idx:end_idx]
                enc_X_batch = self.TDE_encoder.transform(batch_X)

                try:
                    if self.task == "class":
                        y_hat = self.nn_model(torch.Tensor(enc_X_batch).to(self.device))
                        y_hat = torch.argmax(y_hat, dim=1).cpu().numpy()
                    else:
                        y_hat = self.nn_model(torch.Tensor(enc_X_batch).to(self.device))
                        y_hat = y_hat.detach().cpu().numpy()
                except Exception as e:
                    raise RuntimeError(f"Error during prediction: {str(e)}")

                y_hats.append(y_hat)

        predictions = np.concatenate(y_hats)
        # Ensure consistent output shapes
        if self.task == "reg":
            return predictions.reshape(-1, 1)  # Shape: (n_samples, 1)
        else:
            return predictions.reshape(-1)  # Shape: (n_samples,)


class myDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0.001,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, num_classes, drop):
        super(NeuralNet, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.PReLU())
        layers.append(nn.Dropout(drop))

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))

        layers.append(nn.Linear(hidden_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
