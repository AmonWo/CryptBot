import talib as ta


class TALib:
    @staticmethod
    def add_indicators(df):
        df["RSI"] = ta.RSI(df["Close"], timeperiod=14)
        df["ROC"] = ta.ROC(df["Close"], timeperiod=10)
        df["%R"] = ta.WILLR(df["High"], df["Low"], df["Close"], timeperiod=14)
        df["OBV"] = ta.OBV(df["Close"], df["Volume"])
        df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = ta.MACD(df["Close"], fastperiod=12, slowperiod=26,
                                                                 signalperiod=9)
        return df
