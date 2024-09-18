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

            # Calculate monthly returns in percent and value using the 'Close' values
            monthly_returns_pct = last_date['Close'].pct_change().dropna().rename('Returns_pct')
            monthly_returns_pct = monthly_returns_pct.to_frame().reset_index(drop=True)
            monthly_returns_val = last_date['Close'].diff().dropna().rename('Returns_val')
            monthly_returns_val = monthly_returns_val.to_frame().reset_index(drop=True)

            # Assign the 'Open' and 'Close' values for each month
            monthly_returns_pct['Open'] = first_date['Open'].values[1:]  # Skip the first value to match pct_change
            monthly_returns_val['Open'] = first_date['Open'].values[1:]  # Skip the first value to match pct_change
            monthly_returns_pct['Close'] = last_date['Close'].values[1:]  # Skip the first value to match pct_change
            monthly_returns_val['Close'] = last_date['Close'].values[1:]  # Skip the first value to match pct_change

            # Add the 'Date' column
            monthly_returns_pct['Date'] = [pd.Timestamp(year=year, month=month, day=1) for year, month in last_date.index[1:]]
            monthly_returns_val['Date'] = [pd.Timestamp(year=year, month=month, day=1) for year, month in last_date.index[1:]]

            # Add the 'Ticker' column
            monthly_returns_pct['Ticker'] = ticker
            monthly_returns_val['Ticker'] = ticker

            # Reorder columns to have 'Date', 'Ticker', 'Close', 'Returns'
            monthly_returns_pct = monthly_returns_pct[['Date', 'Ticker', 'Close', 'Returns_pct']]
            monthly_returns_val = monthly_returns_val[['Date', 'Ticker', 'Close', 'Returns_val']]

            # Merge the monthly_returns_pct and monthly_returns_val DataFrames on 'Date' and 'Ticker'
            merged_returns = pd.merge(monthly_returns_pct, monthly_returns_val[['Date', 'Ticker', 'Returns_val']], on=['Date', 'Ticker'])

            # Append the results to the all_monthly_returns DataFrame
            all_monthly_returns = pd.concat([all_monthly_returns, merged_returns], ignore_index=True)

        return all_monthly_returns

    @staticmethod
    def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the covariance matrix for 'Returns_val' grouped by the 'Ticker'.
        Args:
            data_frame (pd.DataFrame): A DataFrame with the data.
        Returns:
            pd.DataFrame: A covariance matrix based on 'Returns_val' for each Ticker.
        """
        # Ensure 'Date' column is in datetime format
        #df = df.reset_index()
        #df['Date'] = pd.to_datetime(df['Date'], utc=True)
        #df.set_index('Date', inplace=True)

        # Ensure the DataFrame has at least two different Tickers to calculate covariance
        if df['Ticker'].nunique() < 2:
            raise ValueError("Covariance requires at least two different tickers")
        
        # Pivot the DataFrame to have tickers as columns and 'Returns_val' as values
        pivot_df = df.pivot(index='Date', columns='Ticker', values='Returns_val')

        # Calculate the covariance matrix
        calculated_covariance_matrix = pivot_df.cov()

        return calculated_covariance_matrix