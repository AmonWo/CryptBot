import pandas as pd


class CsvReader:
    @staticmethod
    def read_dataframe(path_to_csv):
        df = pd.read_csv(path_to_csv, index_col="Timestamp")
        return df

    @staticmethod
    def read_train_dataframe(path_to_csv):
        df = pd.read_csv(path_to_csv)
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
        df.drop("Trades", axis=1, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)
        return df

    @staticmethod
    def save_to_csv(df, path, name=None):
        if name is None:
            df.to_csv(path)
        else:
            df.to_csv(path + name)
