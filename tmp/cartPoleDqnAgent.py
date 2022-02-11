import collections
import os
import random
import gym
import numpy as np

import torch
from typing import Deque
from cartPoleDqn import DQN

PROJECT_PATH = os.path.abspath("C:/Users/ti/Dropbox/bfh/cas-ai/labs/14_dqn_pytorch")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "dqn_cartpole.h5")
TARGET_MODEL_PATH = os.path.join(MODELS_PATH, "target_dqn_cartpole.h5")

class Agent:
    def __init__(self, env: gym.Env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape    # observation shapes
        self.actions = self.env.action_space.n                  # number of actions
        # DQN Agent Variables
        self.replay_buffer_size = 50_000            # buffer size                                   (Hyperparameter)
        self.train_start = 1_000                    # Nach so vielen observations wird gestartet    (Hyperparameter)
        self.memory: Deque = collections.deque(     # memory queue
            maxlen=self.replay_buffer_size
        )
        self.gamma = 0.95                           # discount factor
        self.epsilon = 1.0                          # epsilon start value
        self.epsilon_min = 0.01                     # epsilon end value
        self.epsilon_decay = 0.995                  # epsilon decay rate (kann auch 0.999 sein)
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3                   # learning rate
        self.dqn = DQN(
            self.state_shape,                       # input states
            self.actions,                           # output actions
            self.learning_rate                      # learning rate
        )
        self.target_dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate
        )
        self.batch_size = 32

    def get_action(self, state: np.ndarray):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.dqn(state).detach().numpy() )

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=5)                           # 5 last rewards für Überwachung
        best_reward_mean = 0.0

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state =  self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            state = torch.tensor(state)

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                if done and total_reward < 499:
                    reward = -100.0
                self.remember(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state

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
                    break

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.train_start:
            self.epsilon = 1.0 # do not decrease epsilon
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states).astype(np.float32)
        states_next = np.concatenate(states_next).astype(np.float32)

        q_values = self.dqn(states)
        q_values_next = self.target_dqn(states_next)  # forward()

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                # q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])
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
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=250)
    input("Play?")
    agent.play(num_episodes=50, render=True)
