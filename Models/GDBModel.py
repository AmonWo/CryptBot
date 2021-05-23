from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone


class GDBModel:
    """
    ModelClasss
    """

    def __init__(self):
        self.transformer = None
        self.train_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_trans = None
        self.X_test_trans = None
        self.pred_train = None
        self.pred_test = None
        self.plt_legend = []
        self.model = self.create_model()

    def create_model(self):
        # clf_svm = OneVsRestClassifier(SVC(verbose=True, cache_size=1000), n_jobs=-1)

        clf_gdb = GradientBoostingClassifier(
            verbose=True,
            warm_start=False,
            n_estimators=500,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

        return clf_gdb

    def train_model(self):
        self.model.fit(self.X_train_trans, self.y_train)

    def update_model_data(self, X_train, X_test, y_train, y_test, train_data=None):
        self.train_data = train_data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_trans = self.transformer.fit_transform(X_train)
        self.X_test_trans = self.transformer.fit_transform(X_test)
        print('\nUPDATED MODEL DATA SUCCESSFULLY\n')

    def predict_next_snapshot(self):
        self.pred_train = self.model.predict(self.X_train_trans)
        self.pred_test = self.model.predict(self.X_test_trans)

    def print_accuracy(self):
        print(
            '\nMean Test Accuracy: {:.2f}\nPrediction Train Accuracy Score: {:.2f} \nPrediction Test Accuracy Score: {:.2f}'.format(
                self.model.score(self.X_test_trans, self.y_test),
                accuracy_score(self.y_train, self.pred_train),
                accuracy_score(self.y_test, self.pred_test)
            ))

    def print_report(self):
        print(classification_report(self.y_test, self.pred_test, output_dict=False))
        report = classification_report(self.y_test, self.pred_test, output_dict=True)
        return report

    def print_cross_val_score(self):
        scores = cross_val_score(self.model, self.X_test_trans, self.y_test)
        print(scores)
