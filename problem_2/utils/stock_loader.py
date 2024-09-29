def load_or_download_stock_data(ticker, **kwargs):
    import pandas as pd
    import os
    path = os.path.join(os.getcwd(), "data/raw", f"{ticker}.csv")
    data = pd.DataFrame()
    if os.path.exists(path):
        data = pd.read_csv(path, parse_dates=True, index_col='Date')
    else:
        data = download_stock_data(ticker, **kwargs)
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].apply(lambda x: pd.Timestamp(year=x.year, month=x.month, day=x.day))
        data.set_index('Date', inplace=True)
        data.to_csv(path)
    return data['Close']

def download_stock_data(ticker, **kwargs):
    import yfinance as yf
    stock = yf.Ticker(ticker)
    data = stock.history(**kwargs)
    return data