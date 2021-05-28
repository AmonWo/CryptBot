from Trade import CCXT
from Utility import CsvReader
from Utility import ModelMeasure
from Measure import Plotter
from PreProc import CryptoPreprocessor
from Models import GDBModel
from Models import GdbRegModel
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import accuracy_score

# Halbwegs gute Werte: 48
PREDICTION_DAYS = 96


def main():
    # Initialize classes
    ccxt = CCXT.CCXT()
    plotter = Plotter.Plotter()
    cr = CsvReader.CsvReader()
    df = ccxt.get_ohlcv(symbol="BTC/USD", timeframe="1m")
    gdb = GDBModel.GDBModel()
    preproc = CryptoPreprocessor.CryptoPreprocessor(gdb.model)
    mm = ModelMeasure.ModelMeasure()
    # train_data = data between start/enddate, test_data = data since enddate
    train_data, test_data = preproc.select_date_range(df, end_date='2021-03-30')

    print('TrainData:\n', train_data, '\nTestData:\n', test_data)

    # Add timestamps for the following month
    train_data['Prediction'] = np.where(train_data['Close'].shift(-PREDICTION_DAYS) > train_data['Close'], 1, -1)

    # Split train/test data
    X_train, X_test, y_train, y_test = preproc.split_train_test(train_data)

    print('\nX:\n', X_train)
    print('\ny:\n', y_train)

    # Fit + train
    preproc.pipeline.fit(X_train, y_train)

    y_pred = preproc.pipeline.predict(X_test)

    test_data['Prediction'] = preproc.pipeline.predict(test_data)

    mm.plot_importance(gdb)

    cr.save_to_csv(test_data, 'fresh_predict')

    buysell = ccxt.buy_sell(test_data)

    print('Buysell:\n', buysell)

    test_data.loc[:, 'Buy'] = buysell[0]
    test_data.loc[:, 'Sell'] = buysell[1]

    test_data.at[test_data.first_valid_index(), 'Sell'] = np.NaN

    plotter.plot_buy_sell(test_data)

    print(test_data)

    print('Accuracy Score in %: ', accuracy_score(y_test, y_pred, normalize=True) * 100)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
