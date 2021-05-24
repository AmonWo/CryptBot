from Trade import CCXT
from Utility import CsvReader
from Measure import Plotter
from PreProc import CryptoPreprocessor
from Models import GDBModel
from Models import GdbRegModel
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import accuracy_score


PREDICTION_DAYS = 30 * 96


def main():
    # Initialize classes
    ccxt = CCXT.CCXT()
    plt = Plotter.Plotter()
    cr = CsvReader.CsvReader()
    df = cr.read_dataframe('OHLCV/ETHUSD_15.csv')
    gdb_model = GDBModel.GDBModel().model
    preproc = CryptoPreprocessor.CryptoPreprocessor(gdb_model)

    # Add timestamps for the following month
    df['Prediction'] = np.where(df['Close'].shift(-PREDICTION_DAYS) > df['Close'], 1, -1)

    cr.save_to_csv(df)

    # Cut predicted month from training data

    X_train, X_test, y_train, y_test = preproc.split_train_test(df)

    print('\nX:\n', X_train.head())
    print('\ny:\n', y_train.head())

    # Split train/test data
    # X_train, X_test, y_train, y_test = preproc.split_train_test(X=X, y=y)

    # Fit + train
    preproc.pipeline.fit(X_train, y_train)

    y_pred = preproc.pipeline.predict(X_test)

    print('Accuracy Score in %: ', accuracy_score(y_test, y_pred, normalize=True) * 100)

    # gdb_predict = gdb_model.predict(prediction_days_array)

    # print('\nPREDICT:\n', gdb_predict)

    print(df)

    # cr.save_to_csv(df)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
