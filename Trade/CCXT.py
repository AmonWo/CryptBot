import os
import time

import ccxt
import pandas as pd

from Utility import CsvReader

KRAKEN_API = "fSFx/i/+F3lUcBBOa7iVnYWHQfvkJ2q05h/AcpBM5p2MrH8UjNe3CVSY"
KRAKEN_SECRET = "natuv1/mBc9SU6kBdAgGpa9EMY3/0xK/UG8S5zWlwoDOMVhB0EbqT6mdrV6B2//tuA6kEJT/OP5/ml9ZqJimqQ=="
PHEMEX_API = "3a20513b-fcd3-4056-b0d9-eab4a99838ee"
PHEMEX_SECRET = "QN4O20hKdPw12stFE54-L-3UNebEKXqPEoEyR9v9FZNlOTI0ZWRkZC1lYjljLTRkMmQtOTBmNS0yODUyMWE1Y2M1Mzc"
SECONDS_IN_DAY = 60 * 60 * 24
SECONDS_IN_HOUR = 60 * 60
SECONDS_IN_MINUTE = 60


class CCXT:
    def __init__(self):
        print('CCXT Initialized')
        self.exchange = self.connect('kraken')
        self.cr = CsvReader.CsvReader()

    def connect(self, exchange_id):
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': KRAKEN_API,
            'secret': KRAKEN_SECRET
        })
        print('TEST FETCH:\n', exchange.id, exchange.fetch_ticker('BTC/USD'))
        return exchange

    def get_ohlcv(self, symbol, timeframe=None, limit=None):
        if os.path.isfile('OHLCV/Current_OHLCV.csv'):
            x = os.stat('OHLCV/Current_OHLCV.csv')
            seconds = (time.time() - x.st_mtime)
            days = seconds // SECONDS_IN_DAY
            hours = (seconds - (days * SECONDS_IN_DAY)) // SECONDS_IN_HOUR
            minutes = (seconds - (days * SECONDS_IN_DAY) - (hours * SECONDS_IN_HOUR)) // SECONDS_IN_MINUTE

            print("The age of the given file is: {} Days {} Minutes and {} Minutes".format(days, hours,
                                                                                           minutes))
            if days >= 1:
                return self.fetch_ohlcv(symbol, timeframe, limit)
            else:
                print("USING CACHED OHLCV")
                return self.cr.read_dataframe("OHLCV/Current_OHLCV.csv")
        else:
            return self.fetch_ohlcv(symbol, timeframe, limit)

    def fetch_ohlcv(self, symbol, timeframe, limit):
        if self.exchange.has['fetchOHLCV']:
            data = self.exchange.get_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            main_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            main_df.set_index("Timestamp", inplace=True)
            print('OHLCV:\n', )
            print(main_df)
            return main_df
        else:
            print("CAN'T FETCH OHLCV FROM ", self.exchange.id)
