# ACIT4610 Evolutionary Intelligence : Problem 2

{SHORT SUMMARY}

## How to run

To run the project, you will need some setup:

1. Make sure you have a clean / new python environment active.

2. Make sure you are within the correct problem folder: `{...}/ACIT4610_PORTIFOLIO_KMHS/problem_2`.

3. Run the command `pip install -r ./requirements.txt`

4. Additionally, make sure to have a Jupyter Notebook supported editor open.

5. Proceed to open the project, and go into `main.ipynb`.

6. Press the `Run All` icon.

**NB - Make sure you are connected to WiFi. Installing packages and running the code requires WiFi.**

> The code requires Internet Connection because the project fetches the stocks directly from YahooFinance at runtime.

## Data loading

The project fetches the stock data for the specified `Tickers` at runtime into memory through the YahooFinance Python API. This enables us to experiment with different `Tickers` at ease without downloading everything into the computer and git repository.

## Preprocessing

### Currency

We leverage a Python package to convert the currency for each stock into `USD` to solve the problem of currency mismatch from different stock-market brokerages. Norwegian stocks are retrieved with `NOK` as currency while American stocks has the `USD` as currency. Therefore, our default currency is `USD` to normalize the data.

### TBD
