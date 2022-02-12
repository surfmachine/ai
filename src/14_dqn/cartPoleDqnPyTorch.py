
# ---------------------------------------------------------------------------------------------------------------------
# Aufgabe 14: CartPole DQN with PyTorch
# 12.02.2022, Thomas Iten
# ---------------------------------------------------------------------------------------------------------------------

import collections
import os
import random
import gym
import numpy as np
import torch
from torch import nn
from typing import Deque

#
# Global settings
#
PROJECT_PATH = os.path.abspath("D:/dev/workspace/surfmachine/ai/src/14_dqn")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "dqn_cartpole.h5")
TARGET_MODEL_PATH = os.path.join(MODELS_PATH, "target_dqn_cartpole.h5")


# ---------------------------------------------------------------------------------------------------------------------
# DQN Model
# ---------------------------------------------------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, observation_size : int, action_size : int, hidden_size : int=24, learning_rate : float=0.001 ):
        super().__init__()
        # initialize model
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        # initialize loss function and optimizer
        self.loss_fn   = nn.MSELoss() # mean absolute error , mse
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.net(torch.tensor(x).float())

    def update_model(self, other_model):
        self.load_state_dict( other_model.state_dict() )

    def load_model(self, path: str):
        self.net.load_state_dict(torch.load(path))

    def save_model(self, path: str):
        torch.save(self.net.state_dict(), path )

    def fit(self, states: np.ndarray, q_values: np.ndarray):
        pred = self( torch.tensor(states) )
        loss = self.loss_fn(pred, q_values)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ---------------------------------------------------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------------------------------------------------

class Agent:
    """The agend with the DQM model and DQN target model."""

    def __init__(self, env: gym.Env):
        # Environment
        self.env = env
        # Buffer
        self.replay_buffer_size = 100_000           # buffer size                         Hyperparameter
        self.train_start = 2_500                    # start train after n observations    Hyperparameter
        self.memory: Deque = collections.deque(     # memory double-ended-queue
            maxlen=self.replay_buffer_size
        )
        # Agent Parameters
        self.gamma = 0.95                           # discount factor
        self.epsilon = 1.0                          # epsilon start value
        self.epsilon_min = 0.01                     # epsilon end value
        self.epsilon_decay = 0.999                  # epsilon decay rate (kann auch 0.999 sein)
        self.batch_size = 24  # 16                  # batch size
        # DQN Parameters
        self.observation_size = env.observation_space.shape[0]
        self.action_size      = env.action_space.n
        self.hidden_size      = 36 # 24
        self.learning_rate    = 0.001
        # DQN Model
        self.dqn = DQN(
            self.observation_size,                  # input  = number of observations
            self.action_size,                       # output = number of actions
            hidden_size=self.hidden_size,           # hidden = nummber of hidden layer nodes
            learning_rate=self.learning_rate        # learning rate
        )
        # Target DQN Model
        self.target_dqn = DQN(
            self.observation_size,                  # input  = number of observations
            self.action_size,                       # output = number of actions
            hidden_size=self.hidden_size,           # hidden = nummber of hidden layer nodes
            learning_rate=self.learning_rate        # learning rate
        )

    def get_action(self, state: np.ndarray):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.dqn(state).detach().numpy() )

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=4)   # n last rewards for observation
        best_reward_mean = 0.0                              # best reward mean used to trigger update target model

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                if done and total_reward < 499:
                   reward = -50.0
                self.remember(state, action, reward, next_state, done)

                # Replay and decrease expsilon (only from train start on)
                if len(self.memory) > self.train_start:
                    self.replay()
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                # update reward and state
                total_reward += reward
                state = next_state
                # check if game is done
                if done:
                    if total_reward < 500:
                       total_reward += 100.0
                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)

                    if current_reward_mean > best_reward_mean:
                        self.target_dqn.update_model(self.dqn)
                        best_reward_mean = current_reward_mean
                        self.dqn.save_model(MODEL_PATH)
                        self.target_dqn.save_model(TARGET_MODEL_PATH)
                        print(f"New best mean: {best_reward_mean}")
                        if best_reward_mean > 400:
                            return
        print(f"Best reward mean: {best_reward_mean}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states).astype(np.float32)
        states_next = np.concatenate(states_next).astype(np.float32)

        q_values = self.dqn(states)
        with torch.no_grad():
            q_values_next = self.target_dqn(states_next)

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * max(q_values_next[i])

        self.dqn.fit(states, q_values)

    def play(self, num_episodes: int, render: bool = True):
        self.dqn.load_model(MODEL_PATH)
        self.target_dqn.load_model(TARGET_MODEL_PATH)

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                total_reward += reward
                state = next_state
                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


if __name__ == "__main__":
    # initialize random values
    seed = 4711
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # initialize environment
    env = gym.make("CartPole-v1")
    # Train
    agent = Agent(env)
    agent.train(num_episodes=400)
    # Play
    input("Play?")
    agent.play(num_episodes=12, render=True)
