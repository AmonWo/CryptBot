import ccxt
import pandas as pd

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
