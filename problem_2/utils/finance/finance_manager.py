"""_summary_

Raises:
ValueError: _description_
ValueError: _description_

Returns:
_type_: _description_
"""

import pandas as pd

class StockCalculator:

    @staticmethod
    def monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate the monthly returns for the given DataFrame.
        Args:
            data_frame (pd.DataFrame): A DataFrame with the data.
        Returns:
            pd.DataFrame: A DataFrame with the monthly returns.
        """
        # Ensure the DateTime index is in UTC
        df.index = pd.to_datetime(df.index, utc=True)

        # Initialize an empty DataFrame to store the results
        all_monthly_returns = pd.DataFrame()

        # Group by ticker
        tickers_group = df.groupby('Ticker')

        for ticker, df in tickers_group:
            # Group by year and month within each ticker group
            monthly_group = df.groupby([df.index.year, df.index.month])

            # Calculate the first and last values for each month
            first_date = monthly_group.first()
            last_date = monthly_group.last()

            # Calculate monthly returns using the 'Close' values
            monthly_returns = last_date['Close'].pct_change().dropna().rename('Returns')
            monthly_returns = monthly_returns.to_frame().reset_index(drop=True)

            # Assign the 'Open' and 'Close' values for each month
            monthly_returns['Open'] = first_date['Open'].values[1:]  # Skip the first value to match pct_change
            monthly_returns['Close'] = last_date['Close'].values[1:]  # Skip the first value to match pct_change

            # Add the 'Date' column
            monthly_returns['Date'] = [pd.Timestamp(year=year, month=month, day=1) for year, month in last_date.index[1:]]

            # Add the 'Ticker' column
            monthly_returns['Ticker'] = ticker

            # Reorder columns to have 'Date', 'Ticker', 'Open', 'Close', 'Returns'
            monthly_returns = monthly_returns[['Date', 'Ticker', 'Open', 'Close', 'Returns']]

            # Append the results to the all_monthly_returns DataFrame
            all_monthly_returns = pd.concat([all_monthly_returns, monthly_returns], ignore_index=True)

        return all_monthly_returns