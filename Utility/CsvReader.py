import pandas as pd


class CsvReader:
    @staticmethod
    def read_dataframe(path_to_csv):
        df = pd.read_csv(path_to_csv)
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)
        return df

    @staticmethod
    def save_to_csv(df, path):
        df.to_csv(path)
