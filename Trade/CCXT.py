import os
import time

import ccxt
import numpy as np
import pandas as pd

from Utility import CsvReader

SECONDS_IN_DAY = 60 * 60 * 24
SECONDS_IN_HOUR = 60 * 60
SECONDS_IN_MINUTE = 60
TRANSACTION_RATE = 14


class CCXT:

    def __init__(self, exchange="kraken"):
        print('CCXT INITIALIZED\n')
        self.exchange = self.connect(exchange)
        self.cr = CsvReader.CsvReader()

    def connect(self, exchange_id):
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': KRAKEN_API,
            'secret': KRAKEN_SECRET
        })
        # print('TEST FETCH:\n', exchange.id, exchange.fetch_ticker('ETH/USD'))
        return exchange

    def get_current_price(self):
        pd.set_option('display.max_columns', None)
        current_price = pd.DataFrame(self.exchange.fetchTicker('ETH/USD'))
        current_price = current_price[["timestamp", "open", "high", "low", "close", "baseVolume"]]
        current_price.rename(
            columns={"timestamp": "Timestamp", "open": "Open", "high": "High", "low": "Low", "close": "Close",
                     "baseVolume": "Volume"}, inplace=True)
        current_price['Timestamp'] = pd.to_datetime(current_price['Timestamp'], unit='ms')
        current_price.set_index("Timestamp", inplace=True)
        current_price = pd.DataFrame(current_price.head(1))
        print("CURRENT PRICE:\n", current_price, "\n")
        return current_price

    def get_ohlcv(self, symbol, timeframe=None, limit=None, since=None):
        if os.path.isfile("OHLCV/Current_OHLCV-{}.csv".format(timeframe)):
            x = os.stat("OHLCV/Current_OHLCV-{}.csv".format(timeframe))
            seconds = (time.time() - x.st_mtime)
            days = seconds // SECONDS_IN_DAY
            hours = (seconds - (days * SECONDS_IN_DAY)) // SECONDS_IN_HOUR
            minutes = (seconds - (days * SECONDS_IN_DAY) - (hours * SECONDS_IN_HOUR)) // SECONDS_IN_MINUTE

            print("The age of the given file is: {} Days {} Hours and {} Minutes".format(days, hours,
                                                                                         minutes), "\n")
            if days >= 1:
                return self.fetch_ohlcv(symbol, timeframe, limit, since)
            else:
                print("USING CACHED OHLCV\n")
                return self.cr.read_dataframe("OHLCV/Current_OHLCV-{}.csv".format(timeframe))
        else:
            return self.fetch_ohlcv(symbol, timeframe, limit, since)

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        if self.exchange.has['fetchOHLCV']:
            data = self.exchange.fetchOHLCV(symbol=symbol, timeframe=timeframe, limit=limit, since=since)
            main_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            main_df['Timestamp'] = pd.to_datetime(main_df['Timestamp'], unit='ms')
            main_df.set_index('Timestamp', inplace=True)
            self.cr.save_to_csv(main_df, "OHLCV/Current_OHLCV-{}.csv".format(timeframe))
            print('OHLCV:\n', )
            print(main_df)
            return main_df
        else:
            print("CAN'T FETCH OHLCV FROM ", self.exchange.id)

    def buy_sell(self, sold, counter, flag, current_price, buyprice, sellprice, money):
        if sold is True:
            buyprice = 0
            sellprice = 0
            sold = False
        if current_price["Prediction"][0] == 1 and counter == 0:
            if flag != 1:
                buyprice = current_price["Close"][0]
                flag = 1
        elif current_price["Prediction"][0] == -1 and counter == 0:
            if flag != 0:
                sellprice = current_price["Close"][0]
                flag = 0

        if buyprice and sellprice > 0:
            diff = sellprice - buyprice
            sold = True
            print(
                "#####################################################################################################\nBUY: {} SELL : {} DIFF: {}".format(
                    buyprice, sellprice, diff))
            if buyprice < sellprice:
                money = money + (sellprice - buyprice)
                buyprice = 0
                sellprice = 0
                print("BOUGHT FOR: ", buyprice)
            elif buyprice > sellprice:
                money = money - (buyprice - sellprice)
                buyprice = 0
                sellprice = 0
        return sold, flag, buyprice, sellprice, money

    def start_traiding(self, ta, mc, preproc):
        print("START TRADING\n")
        trades = 0
        running = True
        flag = -1
        counter = 0
        buyprice = 0
        sellprice = 0
        money = 1049.75
        sold = False
        while running:
            trading_data = self.fetch_ohlcv('ETH/USD', timeframe="1m", limit=None)
            if len(trading_data) >= 60:
                trading_data = ta.add_indicators(trading_data)
                trading_data = preproc.remove_nans(trading_data)
                current_price = pd.DataFrame(trading_data.iloc[[-1]])
                current_price = mc.predict(current_price, preproc)
                print("TRADING PRICE:\n", current_price)
                sold, flag, buyprice, sellprice, money = self.buy_sell(sold, counter, flag, current_price, buyprice,
                                                                       sellprice, money)
                # self.test_trading(current_data)
                if sold:
                    trades += 1
                time.sleep(60)
                # trading_data.drop(trading_data.head(0).index, inplace=True)
                # trading_data = pd.concat(trading_data, self.get_current_price(), sort=True)
                # print("TRADING_DATA: ", trading_data)
                print(
                    "################################################################################################"
                    "\nTotal Money: {} \nTotal Trades: {}".format(money, trades))

            counter += 1
            if counter == TRANSACTION_RATE:
                counter = 0

    def prepare_buysell(self, df):
        print("TEST DATA:\n", df)
        buysell = self.test_buy_sell(df)

        df.loc[:, 'Buy'] = buysell[0]
        df.loc[:, 'Sell'] = buysell[1]

        df.at[df.first_valid_index(), 'Sell'] = np.NaN
        return df

    def test_trading(self, current_data):

        print("CURRENT DATA:\n", current_data)

        buyprice = 0
        sellprice = 0
        buy_data = np.array(current_data["Buy"])
        sell_data = np.array(current_data["Sell"])
        money = 1000

        for i in range(len(buy_data)):
            if buy_data[i] > 0:
                buyprice = buy_data[i]
            elif sell_data[i] > 0:
                sellprice = sell_data[i]
            if buyprice and sellprice > 0:
                print("BUY: {} SELL : {}".format(buyprice, sellprice))
                if buyprice < sellprice:
                    money = money + (sellprice - buyprice)
                    buyprice = 0
                    sellprice = 0
                elif buyprice > sellprice:
                    money = money - (buyprice - sellprice)
                    buyprice = 0
                    sellprice = 0

        print("Win in Trading Period: ", money, "\n")

    def test_buy_sell(self, df):
        """
        Takes df with Predictions(-1, 1) and appends the Buy/Sell Price
        :param df:
        :return:
        """
        buy_price = []
        sell_price = []
        flag = -1
        counter = 0

        for i in range(len(df)):
            if df['Prediction'][i] > .5 and counter == 0:
                if flag != 1:
                    buy_price.append(df['Close'][i])
                    sell_price.append(np.NaN)
                    flag = 1
                else:
                    buy_price.append(np.NaN)
                    sell_price.append(np.NaN)
            elif df['Prediction'][i] < -.5 and counter == 0:
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
            if counter == TRANSACTION_RATE:
                counter = 0

        return buy_price, sell_price
