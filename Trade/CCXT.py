import ccxt
import pandas as pd
import numpy as np


class CCXT:
    def __init__(self):
        print('CCXT Initialized')
        self.exchange = self.connect('phemex')
        # self.fetch_ohlcv(symbol='ETH/USD', timeframe='1d', limit=10)

    def connect(self, exchange_id):
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': '3a20513b-fcd3-4056-b0d9-eab4a99838ee',
            'secret': 'QN4O20hKdPw12stFE54-L-3UNebEKXqPEoEyR9v9FZNlOTI0ZWRkZC1lYjljLTRkMmQtOTBmNS0yODUyMWE1Y2M1Mzc'
        })
        print(exchange.id, exchange.fetch_ticker('ETH/USD'))
        return exchange

    def fetch_ohlcv(self, symbol, timeframe, limit):
        if self.exchange.has['fetchOHLCV']:
            data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            print('OHLCV\n')
            main_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close' 'Volume', 'Trades'])
            print(main_df)
            return main_df
        else:
            'NO OHLCV'

    def buy_sell(self, df):
        buy_price = []
        sell_price = []
        flag = -1
        counter = 0
        n = 2

        for i in range(len(df)):
            if df['Prediction'][i] == 1 and counter == 0:
                if flag != 1:
                    buy_price.append(df['Close'][i])
                    sell_price.append(np.NaN)
                    flag = 1
                else:
                    buy_price.append(np.NaN)
                    sell_price.append(np.NaN)
            elif df['Prediction'][i] == -1 and counter == 0:
                if flag != 0:
                    buy_price.append(np.NaN)
                    sell_price.append(df['Close'][i])
                    flag = 0
                else:
                    buy_price.append(np.NaN)
                    sell_price.append(np.NaN)
            else:
                buy_price.append(np.NaN)
                sell_price.append(np.NaN)

            counter += 1
            if counter == n:
                counter = 0

        return buy_price, sell_price
