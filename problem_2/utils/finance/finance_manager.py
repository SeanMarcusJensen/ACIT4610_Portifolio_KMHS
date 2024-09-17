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
from currency_converter import CurrencyConverter

class Company:
    """_summary_

    Returns:
        Company: A Company object.
    """
    currency = 'USD'
    __valuta_converter = CurrencyConverter() # Private

    def __init__(self, ticker: fy.Ticker, history_func) -> None:
        self.__ticker = ticker
        self.data_frame: pd.DataFrame = history_func(ticker)
        self.__convert_currency()

        self.__monthly_returns: pd.DataFrame = pd.DataFrame()
    
    def as_dataframe(self) -> pd.DataFrame:
        """ Get the data as a DataFrame. """
        return self.data_frame
    
    def get_monthly_returns(self) -> pd.DataFrame:
        if self.__monthly_returns is None or self.__monthly_returns.empty:
            monthly_df = self.data_frame.reset_index()
            monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
            monthly_df.set_index('Date', inplace=True)

            # Group by year and month
            monthly_group = monthly_df.groupby([monthly_df.index.year, monthly_df.index.month])

            # Calculate the first and last values for each month
            first_date = monthly_group.first()
            last_date = monthly_group.last()


            # Calculate monthly returns using the 'Close' values
            monthly_returns = last_date['Close'].pct_change().dropna().rename('Monthly Returns(%)')
            monthly_returns = monthly_returns.to_frame().reset_index(drop=True)

            # Assign the 'Open' and 'Close' values for each month
            monthly_returns['Open'] = first_date['Open'].values[1:]  # Skip the first value to match pct_change
            monthly_returns['Close'] = last_date['Close'].values[1:]  # Skip the first value to match pct_change

            monthly_returns['Date'] = last_date.index[1:]

            # Add the 'Ticker' column
            monthly_returns['Ticker'] = self.__ticker.ticker

            monthly_returns = monthly_returns[['Date', 'Ticker', 'Open', 'Close', 'Monthly Returns(%)']]


            self.__monthly_returns = monthly_returns

        return self.__monthly_returns

    def __convert_currency(self) -> None:
        __valuta_columns = ['Open', 'High', 'Low', 'Close']
        ticker_currency = self.__ticker.info['currency'] if 'currency' in self.__ticker.info else self.currency
        _currency_conversion = lambda x: self.__valuta_converter.convert(x, ticker_currency, self.currency)
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
        for k, v in self.companies.items():
            data_frame = v.as_dataframe()
            data_frame['Ticker'] = k
            data = pd.concat([data, data_frame], axis=0)
        return data
    
    def get_ticker_count(self) -> int:
        """ Get the number of tickers. """
        return len(self.companies.keys())

    def get(self, key: str) -> Company:
        return self.companies[key]

    def __getitem__(self, key: str) -> pd.DataFrame:
        """ Get the data for a specific ticker. """
        return self.companies[key].as_dataframe()