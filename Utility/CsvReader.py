import pandas as pd
from datetime import datetime


class CsvReader:
    def __init__(self):
        print('CsvReader Initialized')

    def read_dataframe(self, path_to_csv):
        df = pd.read_csv(path_to_csv)
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)
        return df

    def save_to_csv(self, df, name=None):
        if name is None:
            df.to_csv('Export/PredictExport.csv')
        else:
            df.to_csv('Export/' + name + '.csv')
