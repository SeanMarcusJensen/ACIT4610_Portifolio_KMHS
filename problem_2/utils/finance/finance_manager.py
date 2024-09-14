"""_summary_

Raises:
ValueError: _description_
ValueError: _description_

Returns:
_type_: _description_
"""

import yfinance as fy
import datetime
import pandas as pd
import os
from currency_converter import CurrencyConverter

class Company:
    currency = 'USD'
    valuta_converter = CurrencyConverter()

    def __init__(self, ticker: fy.Ticker, history_func) -> None:
        self.__ticker = ticker
        self.data_frame = history_func(ticker)
        self.__convert_currency()
    
    def as_dataframe(self) -> pd.DataFrame:
        """ Get the data as a DataFrame. """
        return self.data_frame

    def __convert_currency(self) -> None:
        __valuta_columns = ['Open', 'High', 'Low', 'Close']
        ticker_currency = self.__ticker.info['currency'] if 'currency' in self.__ticker.info else self.currency
        _currency_conversion = lambda x: self.valuta_converter.convert(x, ticker_currency, self.currency)
        self.data_frame[__valuta_columns] = self.data_frame[__valuta_columns].map(_currency_conversion)

    def __getitem__(self, key: str) -> pd.DataFrame:
        """ Get the data for a specific ticker. """
        return self.data_frame[key]


class TickerManager:

    __accepted_intervals = ['1m', '2m', '5m', '15m', '30m',
                            '1h', '1d', '5d',
                            '1wk', '1mo', '3mo']

    def __init__(self, tickers: list[str],
                 start: datetime.datetime,
                 end: datetime.datetime,
                 interval: str = '1d') -> None:
        """ Create a TickerManager object.

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

        loaded = fy.Tickers(tickers)
        self.companies: dict[str, Company] = {}
        for k in loaded.tickers.keys():
            self.companies[k] = Company(loaded.tickers[k], lambda x: x.history(period=interval, start=start, end=end))

    def get_stock_history(self) -> pd.DataFrame:
        """ Get the stock history for a specific ticker. """
        data = pd.DataFrame()
        for v in self.companies.values():
            data = pd.concat([data, v.as_dataframe()], axis=0)
        return data
    
    def get_ticker_count(self) -> int:
        """ Get the number of tickers. """
        return len(self.companies.keys())
    
    def save_to_csv(self, path: str) -> None:
        """ Save the data to a CSV file. """
        data = self.get_stock_history()
        data.to_csv(path)

    def __getitem__(self, key: str) -> pd.DataFrame:
        """ Get the data for a specific ticker. """
        return self.companies[key].as_dataframe()


""" THIS SECTION IS A SIMPLE DOWNLOADER FOR THE DATA.
MIGHT NOT NEED TO DO THIS, SINCE WE CAN JUST USE HTTP!
"""

class FinanceLoader:
    """ Class to load the finance data from Yahoo Finance.

    Raises:
        ValueError: The interval is not accepted.
    """

    __accepted_intervals = ['1m', '2m', '5m', '15m', '30m',
                            '1h', '1d', '5d',
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
    
    def download(self, path: str) -> TickerManager:
        """ Download the data from Yahoo Finance. """
        self.__create_data_folder_if_not_exists(path)
        data = pd.DataFrame()

        for ticker in self.tickers:
            ticker_data = self.__load(path, ticker)
            if ticker_data.empty is True:
                print(f'Error loading {ticker}, because data is None.')
                continue
            data = pd.concat([data, ticker_data], axis=0)
        
        manager = TickerManager(data, self)
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
