import numpy as np
from tqdm import tqdm 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score


# load PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from .tde import TreeDrivenEncoder


class DeepTFL(BaseEstimator):
    def __init__(
        self, n_est: int = 23, max_depth: int = 4, drop: float = 0.23, xgb_lr: float = 0.5, batchsize: int = 320,
        nn: int = 128, n_layers: int = 4, task: str = 'class', debug: bool = False, checkpoint_name: str = 'checkpoint.pt'
    ):
        """
        A Deep Tree-Boosted Neural Network classifier/regressor.
        """
        self.xgb_model = None
        self.task = task
        self.nn_model = None
        self.TDE_encoder = None
        self.shape = None

        self.n_est = n_est
        self.max_depth = max_depth
        self.nn = nn
        self.drop = drop
        self.debug = debug
        self.xgb_lr = xgb_lr
        self.n_layers = n_layers
        self.checkpoint_name = checkpoint_name
        self.batchsize = batchsize

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        train_data = xgb.DMatrix(X_train, y_train)
        if self.task == 'class':
            gbdt_model = xgb.XGBClassifier
            num_of_outputs = len(set(y_train))
        else:
            gbdt_model = xgb.XGBRegressor
            num_of_outputs = 1

        self.xgb_model = gbdt_model(learning_rate=self.xgb_lr, n_jobs=-1,
                                     max_depth=self.max_depth,
                                     n_estimators=self.n_est,
                                     use_label_encoder=False,
                                     )
        self.xgb_model.fit(X_train, y_train)

        trees = self.xgb_model.get_booster().get_dump(with_stats=False)

        self.TDE_encoder = TreeDrivenEncoder()
        self.TDE_encoder.fit(trees)

        enc_X_train = self.TDE_encoder.transform(X_train)
        self.shape = enc_X_train.shape[1]

        if X_val is not None:
            X_val = self.TDE_encoder.transform(X_val)

        if self.debug:
            print("Shape of the encoded data", enc_X_train.shape)
        self.nn_model = pytorch_train_ann(enc_X_train, y_train, X_val, y_val, self.shape,
                                          self.nn, num_of_outputs,
                                          self.drop, self.n_layers, self.task, self.batchsize, self.checkpoint_name)
        if self.debug:
            print('Training is done')
            
    def predict(self, X: np.ndarray):
        self.nn_model.load_state_dict(torch.load(self.checkpoint_name))

        enc_X_test = self.TDE_encoder.transform(X)
        if self.debug:
            print(enc_X_test.shape)

        if self.task == 'class':
            y_hat = pytorch_predict(self.nn_model, enc_X_test, self.task)
            y_hat = np.rint(y_hat)
            return y_hat
        else:
            if self.debug:
                print('here')
            y_hat = pytorch_predict(self.nn_model, enc_X_test, self.task)
            return y_hat

    
    def predict_proba(self, X: np.ndarray):
        """
        Only for classification problems. Return probability.
        """
        self.nn_model.load_state_dict(torch.load(self.checkpoint_name))
        enc_X_test = self.TDE_encoder.transform(X)

        y_hat = pytorch_predict(self.nn_model, enc_X_test, self.task)
        return y_hat

def pytorch_predict(model, X, task='class'):
    model.eval()
    model.cuda()
    
    # TODO ADD BATCH SIZE 
    
    mydataset = myDataset(np.array(X))
    
    test_loader = torch.utils.data.DataLoader(dataset=mydataset,
                                               batch_size=1000,
                                               shuffle=False)
    out = []
    for samples in test_loader:
        samples = samples.cuda().float()
        if task == 'class':
            y_hat = torch.softmax(model(samples),-1)[:,1]
        else:
            y_hat = model(samples)
        out.append(y_hat.detach().cpu().numpy().flatten())
    return [item for sublist in out for item in sublist]







class myDataset(Dataset):
    '''
    Dataset Class for PyTorch model
    '''

    def __init__(self, X, y=[]):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        if len(self.y) != 0:
            return self.X[index], self.y[index]
        else:
            return self.X[index]

    def __len__(self):
        return len(self.X)


def swish(x):
    return x * torch.sigmoid(x)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, num_classes, drop=0.25):
        super(NeuralNet, self).__init__()
        self.activation = swish
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.layers = nn.ModuleList()
        self.fc4 = nn.Linear(hidden_dim, num_classes)

        self.dropout1 = nn.Dropout(drop)

        # Hidden Layers (number specified by n_layers)
        self.layers.extend([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)])

    def forward(self, x):

        x = self.activation(self.fc1(x))

        for layer in self.layers:
            x = self.dropout1(self.activation(layer(x)))

        x = self.fc4(x)

        return x


def create_data_loader(X, y, batch_size, shuffle):
    dataset = myDataset(X, y)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def train(model, device, train_loader, criterion, optimizer, task):
    model.train()
    for i, (sample, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(sample.float().to(device))
        if task == 'class':
            loss = criterion(outputs, labels.long().to(device))
        else:
            loss = criterion(outputs.reshape(-1), labels.float().to(device).reshape(-1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


def evaluate(model, device, test_loader, criterion, task='class'):
    model.eval()
    test_loss = 0
    correct = 0

    all_outputs, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).long()
            output = model(data)
            if task == 'class':
                test_loss += criterion(output, target).item()  # sum up batch loss
                all_outputs.append(output[:, 1].detach().cpu().numpy())
            else:
                test_loss += criterion(output.reshape(-1), target.float().reshape(-1)).item()  # sum up batch loss
                all_outputs.append(output.detach().cpu().numpy())

            all_targets.append(target.detach().cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_loss *= 100

    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)

    if task == 'class':
        custom_metric = roc_auc_score(all_targets, all_outputs)
    else:
        custom_metric = r2_score(all_targets, all_outputs)

    return test_loss, custom_metric



def pytorch_train_ann(X_train, y_train, X_val=None, y_val=None, input_size=0, hs1=128, num_outs=1, drop=0.25, n_layers=4,
                      task='class', batchsize=1024, path_name='checkpoint.pt'):
    if task == 'class':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, stratify=y_train)

    train_loader = create_data_loader(X_train, y_train, batch_size=batchsize, shuffle=True)
    val_loader = create_data_loader(X_val, y_val, batch_size=batchsize, shuffle=False)

    model = NeuralNet(input_size, hs1, n_layers, num_outs, drop).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=False, path=path_name)

    avg_train_losses = []
    train_losses = []

    num_epochs = 10000

    for epoch in tqdm(range(num_epochs), desc="Epochs: "):
        train(model, device, train_loader, criterion, optimizer, task)

        train_loss, _ = evaluate(model, device, train_loader, criterion, task)
        val_loss, custom_metric = evaluate(model, device, val_loader, criterion, task)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            print('LOSS:', val_loss)
            break

    print('Num epochs:', epoch)

    return model




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
