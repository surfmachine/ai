import random

import numpy as np
import gym
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from dataset import SP500DataSet

class TradingEnv(gym.Env):
    """The S&P 500 Trading Environment."""

    INITIAL_IDX = 0
    INITIAL_CASH = 10_000
    INITIAL_PORTFOLIO_VALUE = 0

    ACTIONS = ["sell", "hold", "buy"]

    def __init__(self, df_train, df_test, play=False):
        # self.df_train = df_train
        # self.df_test = df_test
        # self.play = play
        # self.config = EnvConfig()

        # df and starting index
        self.df = df_test if play else df_train
        self.current_idx = TradingEnv.INITIAL_IDX
        # cash and portfolio
        self.cash = TradingEnv.INITIAL_CASH
        self.portfolio_value = TradingEnv.INITIAL_PORTFOLIO_VALUE
        # target stocks and stock values
        self.stocks = ['AAPL', 'MSFT', 'AMZN', 'NFLX', 'XOM', 'JPM', 'T'] # target stocks
        self.stock_values = np.zeros(len(self.stocks))

        #self.stocks_adj_close_names = [stock + '_Adj_Close' for stock in self.stocks] # AAPL_Adj_Close so heissen die Spalten bei mir ?
        #self.weights = np.full((len(self.config.stocks)), 1 / len(self.config.stocks), dtype=float)
        #self.weights * self.config.initial_cash

        # number, states and rewards
        self.n = len(self.df)
        self.states = self.df.loc[:, ~self.df.columns.isin(self.stocks)].to_numpy()
        self.rewards = self.df[self.stocks].to_numpy()

        # last step data
        self.last_step = None


    def reset(self):
        self.current_idx = TradingEnv.INITIAL_IDX
        self.cash = TradingEnv.INITIAL_CASH
        self.portfolio_value = TradingEnv.INITIAL_PORTFOLIO_VALUE
        self.stock_values = np.zeros(len(self.stocks))
        state = self.states[self.current_idx]
        state = np.array(state).reshape(1, -1)
        # state = torch.tensor(state).float().to(self.config.device)
        self.last_step = None
        return state


    def step(self, action):
        """
        Run the give action, take a step forward and return the next state with the reward and done flag.

        The actions calculates the difference between the mean value of the next states and the current states.

        The reward is then calculated according the following table:

            Action      Difference   Rise    Reward
            sell        positive     True    -10
            sell        negative     False   +10
            buy         positive     True    +20
            buy         negative     False   -10
            hold        n/a          n/a       0

        :param action: ["sell", "hold", "buy"]
        :return: next_state, reward, done
        """

        # check valid state
        if self.current_idx >= self.n:
            raise Exception("Episode already done")

        # check valid actions
        if action not in TradingEnv.ACTIONS:
            raise Exception("Invalid action: " + action)

        # apply action and calculate mean values before and after
        mean = np.mean(self.states[self.current_idx])
        self.current_idx += 1                               # apply action
        next_mean = np.mean(self.states[self.current_idx])

        # calculate done
        done = (self.current_idx == self.n - 1)
        if done:
            next_state = None
            reward = 0
        else:
            # calculate reward
            reward = 0
            rise = (next_mean - mean) > 0

            if action == "sell":
                reward = -10 if rise else +10
            elif action == "buy":
                reward = +20 if rise else -10

            # calculate next step
            next_state = self.states[self.current_idx]
            next_state = np.array(next_state).reshape(1, -1)
            #next_state = torch.tensor(next_state).reshape(1, -1).float().to(self.config.device)

        # save last step data
        self.last_step = {"action": action, "rise": rise, "reward": reward, "done": done}

        # return results
        return next_state, reward, done


    def render(self):
        # Currently we just render the data of the last step


        print(self.last_step["action"] + ":",
              "vaules rised=" + str(self.last_step["rise"]),
              "reward=" + str(self.last_step["reward"]),
              "done=" + str(self.last_step["done"]))

    def render_state_mean_values(self, start=0, n=None):
        means = []
        steps = []
        stop = len(self.states) if n is None else n
        for i in range(start, stop):
            mean = np.mean(self.states[i])
            means.append(mean)
            steps.append(i)
        self.plot(steps, means, title="State values", xlabel="Step", ylabel="Mean")


    def plot(self, x, y, title="", xlabel="", ylabel=""):
        """Simple plot function.
        Further details see: https://jakevdp.github.io/PythonDataScienceHandbook/04.01-simple-line-plots.html
        """
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(x, y);
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


if __name__ == "__main__":

    # init train and test df
    dataset = SP500DataSet()
    df_train, df_test = dataset.get_train_test()

    # init environment and show values
    env = TradingEnv(df_train, df_test)
    env.reset()

    # show the first 100 state mean values
    env.render_state_mean_values(n=100)

    # test some actions
    n=24
    rewards = []

    print("Actions:")
    for _ in range(n):
        action = TradingEnv.ACTIONS[random.randint(0,2)]
        next_state, reward, done = env.step(action)
        rewards.append(reward)
        env.render()

    env.plot(range(0,n), rewards, title="Rewards", xlabel="Step", ylabel="Reward")