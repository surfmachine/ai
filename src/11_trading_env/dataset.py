
import os, os.path, requests, datetime
import numpy as np
import pandas as pd
import bs4 as bs
import yfinance

class SP500DataSet():
    """Data module for the 'Standard and Poor 500' company performance data, adapted for time series support.

    18.01.2021, Thomas Iten

    NOTE:
    To support the TFT task, the time_series flag has be introduced. By enabling this option,
    the data will be enhanced with the a time index and group flag, to support the TimeSeriesDataSet format.
    """

    URL = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(self,
                 train_val_test_split=[85,0,15],
                 spy_binary=False,
                 path=".",
                 force_download=False):
        super().__init__()
        # download properties
        self.start = datetime.datetime(2010, 1, 1)
        self.end = datetime.datetime.now()
        self.fname = path + "/sp500.csv"
        self.force_download = force_download
        # data properties
        self.spy_binary = spy_binary
        self.train_val_test_split = train_val_test_split
        # data
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        # prepare data
        self.load_data()
        self.split_data()


    def load_data(self):
        """Download and prepare the SP500 data."""
        df = self._download_sp500()
        if self.spy_binary:
            df.SPY = [1 if spy > 0 else 0 for spy in df.SPY]
        self.data = df

    def split_data(self):
        """Split data into test, val and train dataframes."""

        train_percent, val_percent, test_percent = self.train_val_test_split
        rows = self.data.shape[0]
        test_rows = int(test_percent * rows / 100)
        val_rows = int(val_percent * rows / 100)
        train_rows =  rows - test_rows - val_rows

        print("Define data splits:")
        print("- Train rows : {0:04d} ( {1}%)".format(train_rows, train_percent))
        print("- Val   rows : {0:04d} ( {1}%)".format(val_rows, val_percent))
        print("- Test  rows : {0:04d} ( {1}%)".format(test_rows, test_percent))
        print("- Total rows : {0:04d} (100%)".format(rows))
        print()

        self.train_data = self.data.iloc[:train_rows]
        self.val_data = self.data.iloc[train_rows:train_rows+val_rows]
        self.test_data = self.data.iloc[train_rows+val_rows:]
        print("Define data shapes:")
        print("- Train shape: {}".format(self.train_data.shape))
        print("- Val   shape: {}".format(self.val_data.shape))
        print("- Test  shape: {}".format(self.test_data.shape))
        print("- Total shape: {}".format(self.data.shape))

    def get_all(self):
        return self.data

    def get_train_test(self) -> pd.DataFrame:
        return self.train_data, self.test_data

    def get_train_val_test(self) -> pd.DataFrame:
        return self.train_data, self.val_data, self.test_data

    def _transform_to_timeseries_df(self, df):
        """Helper class for time series."""
        # create time index
        df['date'] = df.index.to_pydatetime()
        df["time_idx"] = df["date"].dt.year * 12 * 31 + df["date"].dt.month * 31 + df["date"].dt.day
        df["time_idx"] -= int(df["time_idx"].min())
        # create group (At least one groupt is used for the TimeSeriesDataSet)
        df['group_id'] = 0
        # return result
        return df

    def _download_sp500(self) -> pd.DataFrame:
        """Download the SP500 data from the internet, save data to a csv file and return the result.

        Notes:
        - All further calls will serve the data from the csv file.
        - To trigger a new download from the internet, set the force_download flag to True.
        """
        # Load data from file
        if os.path.isfile(self.fname) and not self.force_download:
            print("Load SP500 from file:", self.fname)
            return pd.read_csv(self.fname, index_col=0, parse_dates=True)

        # Download data
        print("Download data from:", SP500DataSet.URL)
        resp = requests.get(SP500DataSet.URL)
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})

        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            tickers.append(ticker)

        tickers = [s.replace('\n', '') for s in tickers]
        data = yfinance.download(tickers, start=self.start, end=self.end)
        df_data = data['Adj Close']

        df_spy = yfinance.download("SPY", start=self.start, end=self.end)
        df_spy = df_spy.loc[:, ['Adj Close']]
        df_spy.columns = ['SPY']

        df = pd.concat([df_data, df_spy], axis=1)

        # Prepare data
        df.dropna(axis=0, how='all', inplace=True)
        print("Dropping columns due to nans > 50%:", df.loc[:, list((100 * (df.isnull().sum() / len(df.index)) > 50))].columns)
        df = df.drop(df.loc[:, list((100 * (df.isnull().sum() / len(df.index)) > 50))].columns, 1)
        df = df.ffill().bfill()
        print("Any columns still contain nans:", df.isnull().values.any())

        df_returns = pd.DataFrame()
        for name in df.columns:
            df_returns[name] = np.log(df[name]).diff()

        df_returns.dropna(axis=0, how='any', inplace=True)

        # Save data and return result
        print("Save data to file:", self.fname)
        df_returns.to_csv(self.fname)

        self.data = df_returns
        return self.data


if __name__ == "__main__":
    dataset = SP500DataSet() # SP500 Dataset
    train_df, test_df = dataset.get_train_test()
