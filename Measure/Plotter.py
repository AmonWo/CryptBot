import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import (
    DateFormatter, AutoDateLocator, AutoDateFormatter, datestr2num
)
from datetime import datetime


class Plotter:
    def __init__(self):
        print('Initialized Plotter')

    def plot_deviance(self, pipeline, X_test, y_test):
        test_score = np.zeros((pipeline.named_steps['classifier'].get_params()['n_estimators'],), dtype=np.float64)
        for i, y_pred in enumerate(pipeline.predict(X_test)):
            test_score[i] = pipeline.loss_(y_test, y_pred)

        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.title('Deviance')
        plt.plot(np.arange(pipeline.get_params()['n_estimators']) + 1, pipeline.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(pipeline.get_params()['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        fig.tight_layout()
        plt.show()

    def plot_prediction(self, pipeline, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        # X_test.sort_index(inplace=True)
        sorted_y_test = y_test.sort_index()
        sorted_pred = pd.DataFrame(data=y_pred, columns=['Prediction'], index=X_test.index.copy())
        sorted_pred.sort_index(inplace=True)
        print('\nX_test Head: \n', X_test.head())
        print('\ny_test Head: \n', y_test.head())
        x_axes_date = datestr2num([
            datetime.fromtimestamp(i).strftime('%Y/%m/%d')
            for i in sorted_pred.index
        ])
        fig = plt.figure(figsize=(10, 10))
        axes = fig.add_subplot(111)
        axes.xaxis_date()
        axes.xaxis.set_major_formatter(DateFormatter("%Y/%m/%d"))
        plt.plot(x_axes_date, sorted_y_test)
        plt.plot(x_axes_date, sorted_pred)
        plt.xlabel("Datum")
        plt.ylabel("Price in USD")
        plt.title("ETH/USD Chart")
        plt.show()
