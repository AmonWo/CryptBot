from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from IPython.display import Image
import pydotplus


class RFModel:
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
        self.pipeline = None
        self.plt_legend = []
        self.model = self.create_model()

    def create_model(self):
        clf_forest = RandomForestClassifier(n_estimators=10,
                                            max_depth=100,
                                            criterion='gini',
                                            # class_weight='balanced',
                                            verbose=2,
                                            oob_score=True,
                                            max_features='auto',
                                            max_samples=500,
                                            bootstrap=True,
                                            warm_start=False,
                                            # min_samples_leaf=2,
                                            # min_samples_split=4,
                                            n_jobs=-1)

        params = clf_forest.get_params()
        for key in params:
            label = key + ': ' + str(params[key])
            self.plt_legend.append(mpatches.Patch(color='red', label=label))

        print('\nCREATED RF MODEL SUCCESSFULLY\n')
        return clf_forest

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
        if hasattr(self.model, 'oob_score_'):
            print(
                '\nOOB Score: {:.2f} \nPrediction Train Accuracy: {:.2f} \nPrediction Test Accuracy: {:.2f} \nMean Test Accuracy: {:.2f}'.format(
                    self.model.oob_score_,
                    accuracy_score(self.y_train, self.pred_train),
                    accuracy_score(self.y_test, self.pred_test),
                    self.model.score(self.X_test_trans, self.y_test)
                )),
        else:
            print(
                'Prediction Train Accuracy: {:.2f} \nPrediction Test Accuracy: {:.2f}'.format(
                    accuracy_score(self.y_train, self.pred_train),
                    accuracy_score(self.y_test, self.pred_test)
                )),

    def print_permutation_importance(self):
        # TODO: Make it work
        result = permutation_importance(self.model, self.X_test_trans, self.y_test, n_repeats=10, random_state=42,
                                        n_jobs=-1)

        feats = {}
        for feature, importance in zip(self.features, result.importances_mean.argsort()):
            feats[feature] = importance

        feats = {k: v for k, v in sorted(feats.items(), key=lambda item: item[1])}

        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T, vert=False)
        fig.tight_layout()
        plt.show()

    def print_tree(self):
        index = 0
        for estimator in self.model.estimators_:
            dot = export_graphviz(estimator)
            graph = pydotplus.graph_from_dot_data(dot)
            Image(graph.create_png())
            graph.write_png("visuals/trees/rf_tree" + str(index) + ".png")
            index += 1

    def print_report(self):
        print(classification_report(self.y_test, self.pred_test, output_dict=False))
        report = classification_report(self.y_test, self.pred_test, output_dict=True)
        return report

    def print_cross_val_score(self):
        scores = cross_val_score(self.model, self.X_test_trans, self.y_test)
        print(scores)

    def print_cross_val_pred(self):
        y_train_pred = cross_val_predict(self.model, self.X_train_trans, self.y_train)
        corr_mat = confusion_matrix(self.y_train, y_train_pred)
        # sn.heatmap(corr_mat, annot=True)
        # plt.show()
        print(corr_mat)
