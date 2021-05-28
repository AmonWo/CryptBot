import pandas as pd


class CsvReader:
    @staticmethod
    def read_dataframe(path_to_csv):
        df = pd.read_csv(path_to_csv, index_col=0)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df

    @staticmethod
    def save_to_csv(df, path):
        df.to_csv(path)
