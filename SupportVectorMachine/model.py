import pickle
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score
from snake import Direction


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
    return np.array(attributes)


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


def gather_all_saves():
    dir_path = r'data/'
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
    saves = gather_all_saves()
    n_samples = 3000
    k = 1/5
    direction = Direction.LEFT
    n_attributes = 10
    X, Y = saves_to_data_samples(saves, n_attributes, n_samples)
    dvd_pnt = int(n_samples * k)
    ns = [1, 10, 100, 1000, 3000]
    acc_scores = {}
    for n in ns:
        svm = SVM(learning_rate=0.00001, n_iters=n, lambda_param=0.0000001)
        print(svm.n_iters)
        score = accuracy_test(X, Y, svm, direction, dvd_pnt)
        print(score)
        acc_scores[n] = score

    print(acc_scores)
