import statistics
import random
import matplotlib.pyplot as plt
import pygame
import torch
import time
import os

from food import Food
from snake import Snake, Direction


def main():
    values = [
        # (0.1, 0.8), (0.1, 0.9), (0.1, 0.99),
        # (0.3, 0.8), (0.3, 0.9), (0.3, 0.99),
        # (0.5, 0.8), (0.5, 0.9), (0.5, 0.99)
        (0.3, 0.99)
    ]
    for params in values:
        pygame.init()
        bounds = (300, 300)
        window = pygame.display.set_mode(bounds)
        pygame.display.set_caption("Snake")

        block_size = 30
        snake = Snake(block_size, bounds)
        food = Food(block_size, bounds)

        is_train = False
        epsilon = params[0]
        discount = params[1]
        learning_rate = 0.2
        if is_train:
            agent = QLearningAgent(block_size, bounds, epsilon=epsilon, discount=discount,
                learning_rate=learning_rate, is_training=True)
            total_episodes = 10000
        else:
            qfunction_path = f'{epsilon}_{discount}/{epsilon}_{discount}.pt'
            agent = QLearningAgent(block_size, bounds, epsilon=0.0, discount=0.0,
                learning_rate=0.0, is_training=False, load_qfunction_path=qfunction_path)
            total_episodes = 100
        scores = []
        rewards = []
        run = True
        pygame.time.delay(1000)
        reward, is_terminal = 0, False
        total_reward_per_episod = 0
        episode, total_episodes = 0, total_episodes
        while episode < total_episodes and run:
            pygame.time.delay(30)  # Adjust game speed, decrease to learn agent faster

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            game_state = {"food": (food.x, food.y),
                        "snake_body": snake.body,  # The last element is snake's head
                        "snake_direction": snake.direction}

            direction = agent.act(game_state, reward, is_terminal)
            reward = -0.1
            is_terminal = False
            snake.turn(direction)
            snake.move()
            reward += snake.check_for_food(food)
            food.update()

            if snake.is_wall_collision() or snake.is_tail_collision():
                pygame.display.update()
                pygame.time.delay(1)
                scores.append(snake.length - 3)
                snake.respawn()
                food.respawn()
                episode += 1
                reward -= 100
                rewards.append(total_reward_per_episod-10)
                total_reward_per_episod = 0
                is_terminal = True

            total_reward_per_episod += reward if not is_terminal else 0

            window.fill((0, 0, 0))
            snake.draw(pygame, window)
            food.draw(pygame, window)
            pygame.display.update()

        print(f"Scores: {scores}")
        # This will create a smoothed mean score per episode plot.
        # I want you to create smoothed mean reward per episode plots, that's how we evaluate RL algorithms!
        if not is_train:
            with open(f"{epsilon}_{discount}/avg_test_score.txt", 'w') as f:
                f.write(f'avg {statistics.mean(scores)} in {len(scores)}\nScores:\n' + str(scores))

        os.makedirs(f"{epsilon}_{discount}", exist_ok=True)
        scores = torch.tensor(scores, dtype=torch.float).unsqueeze(0)
        scores = torch.nn.functional.avg_pool1d(scores, int(total_episodes/20), stride=1)
        plt.plot(scores.squeeze(0))
        plt.savefig(f"{epsilon}_{discount}/mean_score_{'test' if not is_train else 'train'}_{epsilon}_{discount}.png", dpi=1000)
        print("Check out mean_score.png")
        plt.clf()

        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(0)
        rewards = torch.nn.functional.avg_pool1d(rewards, int(total_episodes/20), stride=1)
        plt.plot(rewards.squeeze(0))
        plt.savefig(f"{epsilon}_{discount}/mean_reward_{'test' if not is_train else 'train'}_{epsilon}_{discount}.png", dpi=1000)
        print("Check out mean_reward.png")
        plt.clf()

        agent.dump_qfunction(f"{epsilon}_{discount}/{epsilon}_{discount}.pt")
        pygame.quit()



class QLearningAgent:
    def __init__(self, block_size, bounds, epsilon=0.2, discount=0.99, learning_rate=0.1, is_training=True, load_qfunction_path=None):
        """ There should be an option to load already trained Q Learning function from the pickled file. You can change
        interface of this class if you want to."""
        self.block_size = block_size
        self.bounds = bounds
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = torch.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4))
        # self.Q = torch.zeros((2, 2, 2, 2, 2, 2, 2, 2, 4, 4))
        # self.Q =  torch.zeros((2, 2, 2, 2, 4, 4))
        if load_qfunction_path is not None:
            self.Q = self.load_qfunction(load_qfunction_path)
        self.is_training = is_training
        self.obs = None
        self.action = None

    def act(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        if self.is_training:
            return self.act_train(game_state, reward, is_terminal)
        return self.act_test(game_state, reward, is_terminal)

    def act_train(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        """ Update Q-Learning function for the previous timestep based on the reward, and provide the action for the current timestep.
        Note that if snake died then it is an end of the episode and is_terminal is True. The Q-Learning update step is different."""
        # TODO: There are many hardcoded hyperparameters here, what do they do? Replace them with names.
        new_obs = self.game_state_to_observation(game_state, self.block_size, self.bounds)
        if random.random() < self.epsilon:
            new_action = random.randint(0, 3)
        else:
            new_action = torch.argmax(self.Q[new_obs])

        if self.action:
            if not is_terminal:
                update = reward + self.discount * torch.max(self.Q[new_obs]) - self.Q[self.obs][self.action]
            else:
                update = reward - self.Q[self.obs][self.action]
            self.Q[self.obs][self.action] += self.learning_rate * update

        self.action = new_action
        self.obs = new_obs
        return Direction(int(new_action))

    @staticmethod
    def game_state_to_observation(game_state, block_size, bounds):
        def is_tail_in_location(x, y):
            return any([x==tail_part[0] and y==tail_part[1] for tail_part in gs["snake_body"]])
        gs = game_state
        head_x, head_y = gs["snake_body"][-1][0], gs["snake_body"][-1][1]
        # Is food in x direction
        is_u_f = int(gs["food"][1] < head_y)
        is_r_f = int(gs["food"][0] > head_x)
        is_d_f = int(gs["food"][1] > head_y)
        is_l_f = int(gs["food"][0] < head_x)
        # Is wall collision possible in x direction
        is_u_wc = int(head_y == 0)
        is_r_wc = int(head_x + block_size == bounds[0])
        is_d_wc = int(head_y + block_size == bounds[1])
        is_l_wc = int(head_x == 0)
        # # Is tail collision possible in x direction
        is_u_tc = int(is_tail_in_location(head_x, head_y-block_size))
        is_r_tc = int(is_tail_in_location(head_x+block_size, head_y))
        is_d_tc = int(is_tail_in_location(head_x, head_y+block_size))
        is_l_tc = int(is_tail_in_location(head_x-block_size, head_y))
        return is_u_tc, is_r_tc, is_d_tc, is_l_tc, is_u_wc, is_r_wc, is_d_wc, is_l_wc, is_u_f, is_r_f, is_d_f, is_l_f, gs["snake_direction"].value
        # return is_u_wc, is_r_wc, is_d_wc, is_l_wc, is_u_f, is_r_f, is_d_f, is_l_f, gs["snake_direction"].value
        # return  is_u_f, is_r_f, is_d_f, is_l_f, gs["snake_direction"].value

    def act_test(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        obs = self.game_state_to_observation(game_state, self.block_size, self.bounds)
        action = torch.argmax(self.Q[obs])
        return Direction(int(action))

    def dump_qfunction(self, dump_qfunction_path=None):
        os.makedirs("qfunctions", exist_ok=True)
        if dump_qfunction_path is None:
            dump_qfunction_path = f"qfunctions/qf_{time.strftime('%Y-%m-%d_%H.%M.%S')}.pt"
        torch.save(self.Q, dump_qfunction_path)

    def load_qfunction(self, load_qfunction_path):
        return torch.load(load_qfunction_path)


if __name__ == "__main__":
    main()
