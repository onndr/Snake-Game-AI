import copy
import os
import pickle
import pygame
import time

import torch
from torch import nn

from food import Food
from model import game_state_to_data_sample, train_svm, gather_all_saves, saves_to_data_samples, SVM, MLP, train_mlp, \
    BCDataset, compute_accuracy
from snake import Snake, Direction
from sklearn.svm import SVC
import numpy as np


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    # agent = HumanAgent(block_size, bounds)  # Once your agent is good to go, change this line
    # agent = BehavioralCloningAgent(block_size, bounds)
    agent = MLPAgent(block_size, bounds)

    scores = []
    run = True
    pygame.time.delay(1000)
    while len(scores) < 50:
        pygame.time.delay(10)  # Adjust game speed, decrease to test your agent and model quickly

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state)
        snake.turn(direction)

        snake.move()
        snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(300)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print("Scores ", scores)
    print("Max ", max(scores))
    print("Average ", sum(scores)/len(scores))
    agent.dump_data()
    pygame.quit()


class HumanAgent:
    """ In every timestep every agent should perform an action (return direction) based on the game state. Please note, that
    human agent should be the only one using the keyboard and dumping data. """
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

    def act(self, game_state) -> Direction:
        keys = pygame.key.get_pressed()
        action = game_state["snake_direction"]
        if keys[pygame.K_LEFT]:
            action = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Direction.RIGHT
        elif keys[pygame.K_UP]:
            action = Direction.UP
        elif keys[pygame.K_DOWN]:
            action = Direction.DOWN

        self.data.append((copy.deepcopy(game_state), action))
        return action

    def dump_data(self):
        os.makedirs("data", exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H.%M.%S')
        with open(f"data/{current_time}.pickle", 'wb') as f:
            pickle.dump({"block_size": self.block_size,
                         "bounds": self.bounds,
                         "data": self.data[:-10]}, f)  # Last 10 frames are when you press exit, so they are bad, skip them


class MLPAgent:
    """
    MLPAgent   -   Agent   analogiczny   do   klasy  HumanAgent,   który   wykorzystuje
    wytrenowany perceptron do sterowania wężem. Wagi do sieci musi wczytywać z pliku
    generowanego przez skrypt trenujący
    """
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []
        self.ml = MLP(2, [10, 10, 10, 4], nn.Identity)
        train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(BCDataset(), [0.8, 0.1, 0.1])
        train_mlp(self.ml, train_dataset, validate_dataset, batch_size=20, n_epochs=100)
        print("Accuracy ", compute_accuracy(self.ml, test_dataset))
        self.ml.eval()

    def act(self, game_state) -> Direction:
        input = torch.tensor(game_state_to_data_sample(game_state, self.block_size, self.bounds)).to(torch.float32)
        output = self.ml(input)
        pred_probab = nn.Softmax(dim=0)(output)
        pred = pred_probab.argmax().item()
        return Direction(pred)

    def dump_data(self):
        pass


class BehavioralCloningAgent:
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []
        self.ml = {D: SVM() for D in Direction}
        saves = gather_all_saves()
        length = 0
        for save in saves:
            length += len(save['data'])
        data_samples = saves_to_data_samples(saves, 10, length)
        for dir, svm in self.ml.items():
            train_svm(svm, dir, data_samples)

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        data_sample = game_state_to_data_sample(game_state, self.block_size, self.bounds)
        action = self.make_decision_on_action(data_sample)
        self.data.append((copy.deepcopy(game_state), action))
        return action

    def make_decision_on_action(self, data_sample):
        actions = {}
        for dir, svm in self.ml.items():
            actions[dir] = svm.predict(data_sample.reshape(1, -1))
        preferable = max(actions, key=actions.get)
        return preferable

    def dump_data(*args):
        pass

if __name__ == "__main__":
    main()

