import numpy as np
import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self, state_shape: int, num_actions: int, learning_rate: float):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        # init model
        obs_size = self.state_shape[0]
        hidden_size = 24
        n_actions = self.num_actions
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_fn = nn.CrossEntropyLoss() # mean absolute error , mse
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)


    def forward(self, x):
        return self.net(torch.tensor(x).float())

    def update_model(self, other_model):
        # self.net.load_state_dict( other_model.parameters() )
        osd = other_model.state_dict()
        sd = {}
        sd['0.weight'] = osd['net.0.weight']
        sd['0.bias']   = osd['net.0.bias']
        sd['2.weight'] = osd['net.2.weight']
        sd['2.bias']   = osd['net.2.bias']
        sd['4.weight'] = osd['net.4.weight']
        sd['4.bias']   = osd['net.4.bias']
        self.net.load_state_dict( sd )

    def load_model(self, path: str):
        self.net.load_state_dict(torch.load(path))

    def save_model(self, path: str):
        torch.save(self.net.state_dict(), path )

    def fit(self, states: np.ndarray, q_values: np.ndarray):
        pred = self.net( torch.tensor(states))
        loss = self.loss_fn(pred, q_values)
        # Backpropagation
        self.optimizer.zero_grad()   # fit
        loss.backward()
        self.optimizer.step()


osd = {
    'net.a': "Hello",
    'net.b': "World"
}

for k in osd.keys():
    start = len('net') + 1
    print(k[start:])

print(osd.keys())


