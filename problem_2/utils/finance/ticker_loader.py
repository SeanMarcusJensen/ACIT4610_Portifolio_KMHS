import yfinance as fy
import datetime
import pandas as pd
from currency_converter import CurrencyConverter
import os

class TickerLoader:

    __accepted_intervals = ['1m', '2m', '5m', '15m', '30m',
                            '1h', '1d', '5d',
                            '1wk', '1mo', '3mo']
    
    @staticmethod
    def load(path: str, tickers: list[str], start: datetime.datetime, end: datetime.datetime, interval: str = '1d') -> pd.DataFrame:
        loader = TickerLoader(tickers, start, end, interval)
        return loader(path)

    def __init__(self, tickers: list[str],
                 start: datetime.datetime,
                 end: datetime.datetime,
                 interval: str = '1d') -> None:
        """ Creates a callable object to load all ticker stock data as pandas DataFrame.
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

    def __call__(self, path: str) -> pd.DataFrame:
        """ Download the Tickers from Yahoo Finance into path as csv. """
        self.__create_data_folder_if_not_exists(path)
        data = pd.DataFrame()
        for ticker in self.tickers:
            ticker_data = self.__load(path, ticker)
            if ticker_data.empty is True:
                print(f'Error loading {ticker}, because data is None.')
                continue
            data = pd.concat([data, ticker_data], axis=0)
        
        data.reset_index(inplace=True)
        data.set_index('Date', inplace=True)

        return data
    
    def __create_data_folder_if_not_exists(self, path: str) -> None:
        """ Create the data folder if it does not exist. """
        if not os.path.exists(path):
            print(f'Path {path} does not exist. Creating it.')
            os.makedirs(path)

    def __load(self, path: str, ticker: str) -> pd.DataFrame:
        """ Fetch the data from Yahoo Finance or load it from a file if exist. """
        path = f'{path}/{ticker}.csv'
        try:
            if os.path.exists(path):
                return pd.read_csv(path, parse_dates=True, index_col='Date')
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

        # Fetch the data from Yahoo Finance.
        fetched_ticker = fy.Ticker(ticker)

        # Get the stock history for the ticker.
        ticker_data = fetched_ticker.history(
            period=self.interval, start=self.start_date, end=self.end_date)

        # Convert the currency.
        currency = 'USD'
        valuta_converter = CurrencyConverter()

        __valuta_columns = ['Open', 'High', 'Low', 'Close']
        ticker_currency = fetched_ticker.info['currency'] if 'currency' in fetched_ticker.info else currency
        _currency_conversion = lambda x: valuta_converter.convert(x, ticker_currency, currency)
        ticker_data[__valuta_columns] = ticker_data[__valuta_columns].map(_currency_conversion)

        # Add the ticker to the data.
        ticker_data['Ticker'] = ticker

        ticker_data.reset_index(inplace=True)
        ticker_data.set_index('Date', inplace=True)

        return ticker_data