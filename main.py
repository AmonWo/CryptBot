from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score

from Measure import Plotter
from Models import GDBModel
from Models import ModelController
from PreProc import CryptoPreprocessor
from PreProc import TALib
from Trade import CCXT
from Utility import CsvReader
from Utility import ModelMeasure

# Halbwegs gute Werte: 48
PREDICTION_DAYS = 14


def main():
    # Initialize Utility classes
    ccxt = CCXT.CCXT()
    plotter = Plotter.Plotter()
    cr = CsvReader.CsvReader()
    ta = TALib.TALib()
    gdb = GDBModel.GDBModel()
    mc = ModelController.ModelController()
    # gdb = RFModel.RFModel()
    # gdb = GdbRegModel.GdbRegModel()
    mm = ModelMeasure.ModelMeasure()

    # Load Training and Test DataFrames
    train_data = cr.read_train_dataframe("OHLCV/ETHUSD_1.csv")
    test_data = ccxt.get_ohlcv(symbol="ETH/USD", timeframe="1m", since=1420070400000)

    # row specification

    # Add finance indicators
    train_data = ta.add_indicators(train_data)
    test_data = ta.add_indicators(test_data)

    # Create pipeline and preprocessors
    preproc = CryptoPreprocessor.CryptoPreprocessor(gdb.model, train_data)

    # Cleanup Data
    train_data = preproc.remove_nans(train_data)
    test_data = preproc.remove_nans(test_data)

    cr.save_to_csv(train_data, "Export/", "train_data.csv")
    cr.save_to_csv(test_data, "Export/", "test_data.csv")

    # train_data = data between start/enddate, test_data = data since enddate
    train_data = preproc.select_date_range(train_data, start_date="2018-01-01",
                                           end_date=datetime.today().strftime('%Y-%m-%d'))

    print('TrainData:\n', train_data, '\nTestData:\n', test_data, "\n")

    # Add timestamps for the following month
    train_data['Prediction'] = np.where(train_data['Close'].shift(-PREDICTION_DAYS) > train_data['Close'], 1, -1)

    train_data = preproc.sample_label(train_data)

    # Split train/test data
    X_train, X_test, y_train, y_test = preproc.split_train_test(train_data)

    print('\nX:\n', X_train)
    print('\ny:\n', y_train)

    # Fit + train
    preproc.pipeline.fit(X_train, y_train)
    # mm.drop_feature_importance(preproc.pipeline, X_train, y_train, gdb.model)

    y_pred = preproc.pipeline.predict(X_test)
    y_pred = np.around(y_pred, 0).astype(int)

    ccxt.start_traiding(ta, mc, preproc)

    plotter.plot_buy_sell(test_data)
    # current_data['Prediction'] = np.around(preproc.pipeline.predict(current_data), 0).astype(int)

    print(test_data, "\n")

    print('Accuracy Score in %: ', accuracy_score(y_test, y_pred, normalize=True) * 100, "\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
