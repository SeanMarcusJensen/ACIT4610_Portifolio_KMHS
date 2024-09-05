import yfinance as fy
import datetime
import pandas as pd
import os

class TickerManager:

    __accepted_intervals = ['1m', '2m', '5m', '15m', '30m',
                            '60m', '90m', '1h', '1d', '5d',
                            '1wk', '1mo', '3mo']

    def __init__(self, tickers: list[str],
                 start: datetime.datetime,
                 end: datetime.datetime,
                 interval: str = '1d') -> None:
        """ Create a FinanceManager object.

        Args:
            tickers (list[str]): A list of tickers to be used.
            start (datetime.datetime): Start date to get the data from.
            end (datetime.datetime): End date to get the data from.
            interval (str, optional): _description_. Defaults to '1d'.

        Raises:
            ValueError: The interval is not accepted.
        """

        if interval not in self.__accepted_intervals:
            raise ValueError(f'Interval \'{interval}\' not accepted. Accepted intervals are: {self.__accepted_intervals}')

        self.start_date = start
        self.end_date = end
        self.interval = interval

        loaded = fy.Tickers(tickers)
        self.companies: dict[str, fy.Ticker] = loaded.tickers

        self.get_history = lambda ticker: self.companies[ticker].history(period=self.interval, start=self.start_date, end=self.end_date)

    def get_stock_history(self) -> pd.DataFrame:
        """ Get the stock history for a specific ticker. """
        data = pd.DataFrame()
        for k, v in self.companies.items():
            ticker = self.get_history(k)
            ticker['Ticker'] = k
            data = pd.concat([data, ticker], axis=0)
        return data
    
    def get_ticker_count(self) -> int:
        """ Get the number of tickers. """
        return len(self.companies.keys())

    def __getitem__(self, key: str) -> pd.DataFrame:
        """ Get the data for a specific ticker. """
        if key in self.companies.keys():
            return self.get_history(key)
        else:
            raise KeyError(f'Ticker {key} not found in the data.')

class FinanceManager:
    """ Class to manage the finance data from Yahoo Finance.
        Args:
            data (pandas.DataFrame): The finance data.
    """

    def __init__(self, data: pd.DataFrame, loader: 'FinanceLoader' = None) -> None:
        """ Create a FinanceManager object.

        Args:
            tickers (list[str]): A list of tickers to be used.
            start (datetime.datetime): Start date to get the data from.
            end (datetime.datetime): End date to get the data from.
            interval (str, optional): _description_. Defaults to '1d'.

        Raises:
            ValueError: The interval is not accepted.
        """
        self.finance_data = data
        self.loader = loader

    def as_dataframe(self) -> pd.DataFrame:
        """ Get the data as a DataFrame. """
        return self.finance_data
    
    def __getitem__(self, key: str) -> fy.Ticker:
        """ Get the data for a specific ticker. """
        if key in self.companies.keys():
            return self.companies[key]
        else:
            raise KeyError(f'Ticker {key} not found in the data.')

    def get_ticker_count(self) -> int:
        """ Get the number of tickers. """
        return len(self.finance_data['Ticker'].unique())


class FinanceLoader:
    """ Class to load the finance data from Yahoo Finance.

    Raises:
        ValueError: The interval is not accepted.
    """

    __accepted_intervals = ['1m', '2m', '5m', '15m', '30m',
                            '60m', '90m', '1h', '1d', '5d',
                            '1wk', '1mo', '3mo']

    def __init__(self, tickers: list[str],
                 start: datetime.datetime,
                 end: datetime.datetime,
                 interval: str = '1d') -> None:
        """ Create a FinanceManager object.

        Args:
            tickers (list[str]): A list of tickers to be used.
            start (datetime.datetime): Start date to get the data from.
            end (datetime.datetime): End date to get the data from.
            interval (str, optional): _description_. Defaults to '1d'.

        Raises:
            ValueError: The interval is not accepted.
        """
        if interval not in self.__accepted_intervals:
            raise ValueError(f'Interval \'{interval}\' not accepted. Accepted intervals are: {self.__accepted_intervals}')
        
        self.tickers = tickers
        self.start_date = start
        self.end_date = end
        self.interval = interval
        self.storage_path = None
    
    def download(self, path: str) -> FinanceManager:
        """ Download the data from Yahoo Finance. """
        self.__create_data_folder_if_not_exists(path)
        data = pd.DataFrame()

        for ticker in self.tickers:
            ticker_data = self.__load(path, ticker)
            if ticker_data.empty is True:
                print(f'Error loading {ticker}, because data is None.')
                continue
            data = pd.concat([data, ticker_data], axis=0)
        
        manager = FinanceManager(data, self)
        return manager
    
    def __create_data_folder_if_not_exists(self, path: str) -> None:
        """ Create the data folder if it does not exist. """
        if not os.path.exists(path):
            print(f'Path {path} does not exist. Creating it.')
            os.makedirs(path)

    def __load(self, path: str, ticker: str) -> pd.DataFrame:
        """ Fetch the data from Yahoo Finance. """
        path = f'{path}/{ticker}.csv'

        try:
            if os.path.exists(path):
                return pd.read_csv(path)

            data = self.__fetch(ticker)
            if self.__save(path, data):
                return data
            print(f'Error loading {ticker} data.')
            return data
        except Exception as e:
            data = self.__fetch(ticker)
            if data.empty is False:
                if not self.__save(path, data):
                    print(f'Error saving {ticker} data.', e)
            return data

    def __save(self, path: str, data: pd.DataFrame) -> bool:
        """ Save the data to a file. """
        try:
            data.to_csv(path)
            return True
        except Exception as e:
            print(f'Error saving data.', e)
            return False
    
    def __fetch(self, ticker: str) -> pd.DataFrame:
        """ Fetch the data from Yahoo Finance. """
        print(f'Fetching data for {ticker}...')
        fetched_ticker = fy.Ticker(ticker)
        ticker_data = fetched_ticker.history(
            period=self.interval, start=self.start_date, end=self.end_date)
        ticker_data['Ticker'] = ticker
        return ticker_data
