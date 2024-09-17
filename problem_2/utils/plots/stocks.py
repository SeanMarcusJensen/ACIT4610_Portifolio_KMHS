import pandas as pd
import matplotlib.pyplot as plt

class StockPlotter:

    @staticmethod
    def stock_history(dataframe: pd.DataFrame, title: str, column: str) -> None:
        # Ensure 'Date' column is in datetime format
        df = dataframe.reset_index()
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)

        # Group by 'Ticker'
        tickers = df.groupby('Ticker')

        # Plotting
        plt.figure(figsize=(9, 6))
        for ticker, data in tickers:
            plt.plot(data.index, data[column], label=ticker)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price in ($ USD)')
        plt.legend()
        plt.show()