import Trade.CCXT as CCXT
import Utility.CsvReader as CsvReader
import Measure.Plotter as Plotter
import PreProc.CryptoPreprocessor as CryptoPreprocessor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import accuracy_score
from Models import GdbRegModel


def main():
    # Use a breakpoint in the code line below to debug your script.
    ccxt = CCXT.CCXT()
    plt = Plotter.Plotter()
    cr = CsvReader.CsvReader()
    df = cr.read_dataframe('OHLCV/ETHUSD_15.csv')
    gdb_model = GdbRegModel.GdbRegModel().model
    preproc = CryptoPreprocessor.CryptoPreprocessor(df, gdb_model)

    X_train, X_test, y_train, y_test = preproc.split_train_test(df)
    preproc.pipeline.fit(X_train, y_train)
    print('DF Head: \n', df.head())
    # plt.plot_deviance(preproc.pipeline, X_test, y_test)
    plt.plot_prediction(preproc.pipeline, X_test, y_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
