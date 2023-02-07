import pickle
import os
import re
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


class NBC:
    def __init__(self):
        # prior = {class1:cl1_prob,class2:cl2_prob,..}
        # posteriors = {class1:{feature1:{value1:v1_prob,value2:v2_prob,..},..},..}
        self.priors = {}
        self.posteriors = {}

    def fit(self, X, Y):
        # set prior and posterior probs
        classes = set(Y)
        n_observetions = len(Y)
        posts = self.posteriors
        for cl in classes:
            self.priors[cl] = Y.count(cl)/n_observetions
        for i, x in enumerate(X):
            posts[Y[i]] = posts[Y[i]] if posts.get(Y[i]) else {}
            cl_fs = posts[Y[i]]
            for j, value in enumerate(x):
                cl_fs[j] = {value: 1} if not cl_fs.get(j) else {
                    **cl_fs[j], value: cl_fs[j][value] + 1
                    if cl_fs[j].get(value) else 1}

        for cl, features in self.posteriors.items():
            for vals in features.values():
                count = sum(vals.values())
                for v, c in vals.items():
                    vals[v] = c/count

    def predict(self, x):
        # P(A|B) = P(B|A) * P(A) / P(B), this is the equation of Bayes Theorem.
        # no P(B) in this case, because it's the same for each class
        # probs[class] = self.prior[class] * Π(P(xi|class))
        # return most probable class
        probs = {}
        for cl in self.priors:
            probs[cl] = self.priors[cl]
            for i, v in enumerate([val for val in self.posteriors[cl].values()]):
                probs[cl] *= v[x[i]] if v.get(x[i]) else 1
        return max(probs, key=probs.get)

    def accuracy(self, X, Y):
        preds = [self.predict(x) for x in X]
        return len([0 for i in range(0, len(Y)) if Y[i] == preds[i]]) / len(Y)


def game_state_to_data_sample(game_state: dict, block_size, bounds) -> list:
    def is_obstacle_in_location(x, y):
        return any([x==tail_part[0] and y==tail_part[1] for tail_part in gs["snake_body"]])
    gs = game_state
    head_x, head_y = gs["snake_body"][-1][0], gs["snake_body"][-1][1]
    # Is food in x direction
    is_u_f = int(gs["food"][1] < head_y)
    is_r_f = int(gs["food"][0] > head_x)
    is_d_f = int(gs["food"][1] > head_y)
    is_l_f = int(gs["food"][0] < head_x)
    # Is collision possible in x direction
    is_u_c = int(head_y == 0) or int(is_obstacle_in_location(head_x, head_y-block_size))
    is_r_c = int(head_x + block_size == bounds[0]) or int(is_obstacle_in_location(head_x+block_size, head_y))
    is_d_c = int(head_y + block_size == bounds[1]) or int(is_obstacle_in_location(head_x, head_y+block_size))
    is_l_c = int(head_x == 0) or int(is_obstacle_in_location(head_x-block_size, head_y))

    return is_u_c, is_r_c, is_d_c, is_l_c, is_u_f, is_r_f, is_d_f, is_l_f, gs["snake_direction"].value


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


def saves_to_data_samples(saves):
    X = []
    Y = []
    for save in saves:
        for data in save['data']:
            state, action = data
            x = game_state_to_data_sample(state, save['block_size'], save['bounds'])
            X.append(x)
            Y.append(action.value)
    return X, Y


if __name__ == "__main__":
    saves = gather_all_saves()
    X, Y = saves_to_data_samples(saves)
    accuracies = []
    for i in range(50):
        nbc = NBC()
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
        nbc.fit(x_train, y_train)
        accuracies.append(nbc.accuracy(x_test, y_test))
    print(accuracies)
    print("mean accuracy: ", mean(accuracies))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
    bernoulli_nb = BernoulliNB()
    bernoulli_nb.fit(x_train, y_train)
    preds = bernoulli_nb.predict(x_test)
    accuracy = len([0 for i in range(0, len(y_test)) if y_test[i] == preds[i]]) / len(y_test)
    print('Accuracy bernoulli_nb: ', accuracy)

