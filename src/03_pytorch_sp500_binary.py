
# =====================================================================================================================
# Stock exchange data downlaod and experiments
# 14.11.2021, Thomas Iten
# =====================================================================================================================

import os, os.path, pickle, time
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
from functools import reduce

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.utils.data as data_utils
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import bs4 as bs
import requests
import yfinance as yf
import datetime

# =====================================================================================================================
# Load and prepare data
# =====================================================================================================================

class DataHandler():
    """Load 'Standard and Poor 500' companies performace data and split into train- and test-dataset"""

    def __init__(self):
        self.url   = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.start = datetime.datetime(2010, 1, 1)
        self.stop  = datetime.datetime.now()
        self.Ntest = 1000
        self.now   = time.time()
        # data files
        self.path = "data/"
        self.stocks_fname = self.path + "sp500_closefull.csv"        # Standard and Poor 500 companies
        self.train_fname  = self.path + "sp500_train.pickle"         # Training data
        self.test_fname   = self.path + "sp500_test.pickle"          # Testdata

    def load_datasets(self) -> pd.DataFrame:
        if os.path.isfile(self.train_fname) and os.path.isfile(self.test_fname):
            train_data = pickle.load(open(self.train_fname, 'rb'))
            test_data  = pickle.load(open(self.test_fname, 'rb'))
        else:
            train_data, test_data = self.download_data()

        # show results
        print("train_data shape:", train_data.shape)
        print("test_data  shape :", test_data.shape)

        print("\ntrain_data head:")
        print(train_data.head(5))

        print("\ntest_data head:")
        print(test_data.head(5))

        # return result
        return train_data, test_data

    def download_data(self) -> pd.DataFrame:

        # Download 'Standard and Poor 500' companies and save to CSV (once)
        if not os.path.isfile(self.stocks_fname):
            resp = requests.get(self.url)
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            table = soup.find('table', {'class': 'wikitable sortable'})
            tickers = []

            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                tickers.append(ticker)

            tickers = [s.replace('\n', '') for s in tickers]
            data = yf.download(tickers, start=self.start, end=self.end)
            data['Adj Close'].to_csv(self.stocks_fname)

        # Read companies and add SPY
        df0 = pd.read_csv(self.stocks_fname, index_col=0, parse_dates=True)

        df_spy = yf.download("SPY", start=self.start, end=self.end)
        df_spy = df_spy.loc[:, ['Adj Close']]
        df_spy.columns = ['SPY']

        df0 = pd.concat([df0, df_spy], axis=1)

        # Prepare data
        df0.dropna(axis=0, how='all', inplace=True)
        print("Dropping columns due to nans > 50%:", df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)
        df0 = df0.drop(df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns, 1)
        df0 = df0.ffill().bfill()
        print("Any columns still contain nans:", df0.isnull().values.any())

        df_returns = pd.DataFrame()
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()

        df_returns.dropna(axis=0, how='any', inplace=True)

        # Split data into train and test data
        train_data = df_returns.iloc[:-self.Ntest]
        test_data = df_returns.iloc[-self.Ntest:]

        # save data files
        pickle.dump(train_dataset, open(self.train_fname, "wb"))
        pickle.dump(test_dataset,  open(self.test_fname, "wb"))

        # return results
        return train_data, test_data


    def prepare(self, dataset, device, spy_binary=True, batch_size=64):
        """Prepare train- and testdata for the model training and validation"""

        # Convert spy to binary value 0/1
        if spy_binary:
            dataset.SPY = np.where(dataset.SPY >=0, 1, 0)
            print("Convert spy to binary value 0/1:")
            print(dataset.head(5))

        # Split labels and features
        labels = dataset.SPY.values
        features = dataset.iloc[:, :-1].values
        print("\nSplit labels and features:")
        print("- label shape    :", labels.shape)
        print("- features shape :", features.shape)

        # Convert to tensor
        tensor_labels   = torch.tensor(labels).float().to(device)
        tensor_features = torch.tensor(features).float().to(device)


        # Create tensor dataloader
        print("\nCreate tensor dataloader with batch_size={}".format(batch_size))
        data = data_utils.TensorDataset(tensor_features, tensor_labels)
        dataloader = DataLoader(data, batch_size=batch_size)

        # return result
        return dataloader

    def printTitle(self, title):
        print("\ntitle\b")

# ---------------------------------------------------------------------------------------------------------------------

print("\n--- INITIALIZATION -----------------------------------------------------------------------\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
spy_binary = True
batch_size = 10
learning_rate=1e-3
epochs = 3

data_handler = DataHandler()

print("Device        :", device)
print("SPY binary 0/1:", spy_binary)
print("Batch size    :", batch_size)
print("Learning rate :", learning_rate)
print("Epochs        :", epochs)

print("\n--- LOAD DATA ----------------------------------------------------------------------------\n")
train_dataset, test_dataset = data_handler.load_datasets()

print("\n--- PREPARE TRAIN DATA -------------------------------------------------------------------\n")
train_dataloader = data_handler.prepare\
    (train_dataset, device, spy_binary=spy_binary, batch_size=batch_size)

print("\n--- PREPARE TEST DATA --------------------------------------------------------------------\n")
test_dataloader = data_handler.prepare\
    (test_dataset, device, spy_binary=spy_binary, batch_size=batch_size)

# =====================================================================================================================
# Define model (describe forward path)
# =====================================================================================================================

class NeuralNetwork(nn.Module):                 # 1. Klasse erstellen von nn.Module

    def __init__(self):                         # 2. Konstruktor
        super(NeuralNetwork, self).__init__()   # Super Konstruktor aufrufen
        self.linear_relu_stack = nn.Sequential( # Layer 2..n, einfache Netzwerk übereinander (Sequential)
            nn.Linear(489, 512),                # Input 28x28, Output = 512
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(512, 64),                 # Input 512 (dito Output von oben)
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(64, 32),                  # Input 512 (dito Output von oben)
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(32, 1)                    # Prediction für 1 Kategorie
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# =====================================================================================================================
# Train and test model
# =====================================================================================================================

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.unsqueeze(1).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.unsqueeze(1).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = (pred>0.5).float()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg test loss: {test_loss:>8f}")


print("\n--- DEFINE MODEL, LOSS AND OPTIMZER ------------------------------------------------------\n")

model = NeuralNetwork().to(device)      # Angabe wo das ausgeführt werden soll
print(model)

# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # step, update von  params = params - lr * grad


print("\n--- TRAIN AND TEST MODEL -----------------------------------------------------------------\n")

for e in range(epochs):
    print(f"Epoch {e+1}\n------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    print("Done!")
