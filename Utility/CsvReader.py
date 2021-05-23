import pandas as pd


class CsvReader:
    def __init__(self):
        print('CsvReader Initialized')

    def read_dataframe(self, path_to_csv):
        df = pd.read_csv(path_to_csv, index_col=0)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
        return df
