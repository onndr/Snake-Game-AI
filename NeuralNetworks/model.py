import json
import pickle
import os
import re
import time
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch import nn
import torch
from torchmetrics import Accuracy



def game_state_to_data_sample(game_state: dict, block_size, bounds) -> list:
    """
    arguments:
        x0 - current head direction
        x1 - snake length
        x2 - distance from head to top wall
        x3 - distance from head to bottom wall
        x4 - distance from head to left wall
        x5 - distance from head to right wall
        x6 - is food right
        x7 - is food left
        x8 - is food up
        x9 - is food down
    """
    food, snake_body, snake_direction = game_state.values()
    snake_head = snake_body[-1]
    head_x, head_y = snake_head
    food_x, food_y = food
    x_bounds, y_bounds = bounds
    x0 = snake_direction.value
    x1 = len(snake_body)
    x2 = head_x / block_size
    x3 = (x_bounds - head_x - 30) / block_size
    x4 = head_y / block_size
    x5 = (y_bounds - head_y - 30) / block_size
    x6 = 2 if food_x > head_x else -2
    x7 = 2 if food_x < head_x else -2
    x8 = 2 if food_y < head_y else -2
    x9 = 2 if food_y > head_y else -2
    attributes = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
    # attributes = [x6, x7, x8, x9]
    return np.array(attributes)


class BCDataset(Dataset):
    """
    BCDataset - Klasa dziedzicząca po torch.utils.data.Dataset, należy nadpisać jej metody
    __init__   ,   __get_item__,   __len__  https://pytorch.org/docs/stable/data.html#map-
    style-datasets. Ma ona za zadanie zaopatrywać ładowarkę danych (ang. DataLoader)
    w przykłady danych.
    """
    def __init__(self, saves_dir=r'data/', n_attributes=10):
        data_saves = gather_all_saves(saves_dir)
        length = 0
        for save in data_saves:
            length += len(save['data'])
        self.X, self.Y = saves_to_data_samples(data_saves, n_attributes, length)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y

    def __len__(self):
        return self.X.shape[0]


class MLP(Module):
    """
    MLP   (ang.   Multi   Layer   Perceptron)   -   Perceptron   wielowarstowy,   pochodna
    torch.nn.Module.   Klasyfikator,   który   dla   wektora   opisującego   stan   gry   zwróci
    kierunek/akcję węża: góra, prawo, dół, bądź lewo.
    """
    def __init__(self, n_hidden_layers, n_neurons, activation_fun):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(n_neurons)-1):
            self.layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
            if i < n_hidden_layers:
                self.layers.append(activation_fun())

    def forward(self, x):
        return self.layers(x)



"""
Metryki - Należy raportować wartości funkcji kosztu oraz dokładność klasyfikacji na
zbiorze     treningowym   oraz   walidującym   dla   każdej   epoki.   Można   użyć   bibliotek:
logging, Tensorboard, torch.metrics.
"""

def save_metrics(epochs_metrics):
    os.makedirs("mlp_epochs_data", exist_ok=True)
    current_time = time.strftime('%Y-%m-%d_%H.%M.%S')
    with open(f"mlp_epochs_data/{current_time}.json", 'w') as f:
        json.dump(epochs_metrics, f)


def train_mlp(mlp: MLP, train_dataset, validate_dataset, learning_rate=0.01, batch_size=20, n_epochs=5, norm_flag=False):
    """
        Skrypt trenujący - Powinien dokonywać wsadowego trenowania modelu (batch size >
        1) korzystając z ładowarek danych i integrować powyższe moduły. Jako optymalizator
        należy wykorzystać optymalizator SGD. Skrypt powinien generować plik zawierający
        wagi modelu (tzw.  state dict) z najlepszej epoki, tj. epoki, na której model uzyskał
        najwyższą dokładność na zbiorze walidującym.
    """
    if batch_size <= 1:
        batch_size = 2

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate, weight_decay=1e-5)
    results = {}
    current_time = time.strftime('%Y-%m-%d_%H.%M.%S')

    for epoch in range(n_epochs):
        mlp.train()
        current_loss = 0.0
        epoch_accuracies = []
        best_validate_acc = 0.0
        for i, data in enumerate(train_data_loader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            targets = targets.type(torch.LongTensor)

            # m = nn.Dropout(p=0.1)
            # inputs = m(inputs.to(torch.float32))

            outputs = mlp(inputs.to(torch.float32))
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            accuracy = Accuracy(task="multiclass", num_classes=4)
            pred_probab = nn.Softmax(dim=1)(outputs)
            pred = (pred_probab.argmax(1))
            epoch_accuracies.append(accuracy(pred, targets))

        # if norm_flag and epoch == 0:
        #     norms = []
        #     for s in mlp.parameters():
        #         if len(s.size()) > 1:
        #             norms.append("%.6f" % torch.linalg.matrix_norm(s.grad).item())
        #     print(len(norms))
        #     print(norms)

        train_acc = sum(epoch_accuracies) / len(epoch_accuracies)
        validate_acc = compute_accuracy(mlp, validate_dataset)
        results[epoch] = {'train_accuracy': train_acc.item(),
                          'validate_accuracy': validate_acc.item(),
                          'loss': current_loss}
        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            torch.save(mlp.state_dict(), 'model_weights.pth')
    save_metrics(results)
    mlp.load_state_dict(torch.load('model_weights.pth'))


def compute_accuracy(model: MLP, dataset):
    model.eval()
    data_loader = DataLoader(dataset)
    curr_acc = []
    for data in data_loader:
        inputs, targets = data
        targets = targets.type(torch.LongTensor)
        outputs = model(inputs.to(torch.float32))
        accuracy = Accuracy(task="multiclass", num_classes=4)
        pred_probab = nn.Softmax(dim=1)(outputs)
        pred = (pred_probab.argmax(1))
        curr_acc.append(accuracy(pred, targets))
    acc = sum(curr_acc) / len(curr_acc)
    return acc


class SVM:
    def __init__(self, n_iters=1000, lambda_param=0.01, learning_rate=0.001):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = 0

    def fit(self, X, Y):
        n_attributes = len(X[0])
        self.w = np.zeros(n_attributes)
        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                condition = Y[i] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = self.lr * (2 * self.lambda_param * self.w)
                    self.w -= dw
                else:
                    dw = self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, Y[i]))
                    self.w -= dw
                    db = self.lr * Y[i]
                    self.b -= db

    def predict(self, X):
        return np.dot(X, self.w) - self.b


def load_pickle_save(file_path):
    with open(file_path, 'rb') as f:
        save = pickle.load(f)
        return save


def gather_all_saves(dir_path=r'data/'):
    saves_data = []
    for path in os.listdir(dir_path):
        file_path = os.path.join(dir_path, path)
        if os.path.isfile(file_path) and re.match(r'data\/[0-9,\-,\.,_]*\.pickle', file_path):
            saves_data.append(load_pickle_save(file_path))
    return saves_data


def saves_to_data_samples(saves, n_params, n_samples=3000):
    X = np.ndarray((n_samples, n_params), dtype=float)
    Y = np.ndarray((n_samples,), dtype=int)
    X.fill(0)
    Y.fill(0)
    count = 0
    for save in saves:
        for data in save['data']:
            state, action = data
            x = game_state_to_data_sample(state, save['block_size'], save['bounds'])
            for k, a in enumerate(x):
                X.itemset((count, k), a)
            Y.itemset((count), action.value)
            count += 1
            if count == n_samples:
                return X, Y
    return X, Y


def train_svm(svm, direction, data_samples):
    X, Y = data_samples
    Y = np.where(Y == direction.value, 1, -1)
    svm.fit(X, Y)


def accuracy_test(X, Y, svm, direction, dvd_pnt):
    Y = np.where(Y == direction.value, 1, -1)
    X_train, Y_train = X[dvd_pnt:], Y[dvd_pnt:]
    X_test, Y_test = X[:dvd_pnt], Y[:dvd_pnt]
    Y_pred = np.ndarray((len(X_test),), dtype=int)

    svm.fit(X_train, Y_train)
    for i in range(len(X_test)):
        prediction = svm.predict(X_test[i].reshape(1, -1))
        Y_pred.itemset((i), np.sign(prediction))
    return accuracy_score(Y_test, Y_pred)


if __name__ == "__main__":
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(BCDataset(), [0.8, 0.1, 0.1])
    # funcs = [nn.Identity, nn.ReLU, lambda: nn.LeakyReLU(0.01), nn.Tanh]
    # layers = [1, 2, 5, 30]
    funcs = [nn.ReLU]
    layers = [1]
    for func in funcs:
        for i in layers:
            norm_flag = True if func == nn.ReLU and i == 30 else False
            # neurons = [10] + [10 for j in range(i)] + [4]
            # mlp = MLP(i, neurons, func)
            # train_mlp(mlp, train_dataset, validate_dataset, 0.01, 20, 5, norm_flag)
            # print("Activation fun: ", func.__name__)
            # print("Layers number: ", i)
            # print("Test accuracy: ", compute_accuracy(mlp, test_dataset), "\n")
            n_neuron = [10, 8, 6, 4]
            for j in n_neuron:
                neurons = [10, j, 4]
                mlp = MLP(i, neurons, func)
                train_mlp(mlp, train_dataset, validate_dataset, 0.01, 20, 5, norm_flag)
                print("Activation fun: ", func.__name__)
                print("Layers number: ", i)
                print("Neurons: ", j)
                print("Test accuracy: ", compute_accuracy(mlp, test_dataset), "\n")
