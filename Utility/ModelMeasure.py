import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trades']


class ModelMeasure:
    def plot_importance(self, gdb):
        feats = {}
        for feature, importance in zip(FEATURES, gdb.model.feature_importances_):
            feats[feature] = importance * 100

        feats = {k: v for k, v in sorted(feats.items(), key=lambda item: item[1])}

        plt.figure(figsize=(20, 15))
        plt.bar(range(len(feats)), list(feats.values()), align='center')
        plt.legend(handles=gdb.plt_legend)
        plt.axhline()
        plt.xticks(range(len(feats)), list(feats.keys()), rotation=45, fontsize=20, horizontalalignment='right')
        plt.yticks(fontsize=30)
        plt.ylabel('Importance in %', fontsize=20)
        plt.tight_layout()
        plt.savefig('importance.png')
        plt.show()

    @staticmethod
    def drop_feature_importance(pipeline, X_train, y_train, model_instance, scaler='min_max'):
        """
        Removes features iteratively to measure the impact of the feature
        :param pipeline: Pipeline object
        :param X_train: Training data without labels
        :param y_train: Training labels
        :param model_instance: Model instance do train and predict with
        :param scaler: min_max or std - Decides on the scaler used
        """
        benchmark_score = pipeline.score(X_train, y_train)
        feats = {}
        print('\nX_TRAIN_COLS:\n', X_train.columns)
        for index in range(len(X_train.columns)):

            numerical_columns = X_train.columns[X_train.dtypes != 'object'].tolist()

            new_X_train = X_train.drop(X_train.columns[index], axis=1, inplace=False)

            numerical_columns.pop(index)

            num_transformer = None

            if scaler == 'std':
                num_transformer = Pipeline(steps=[
                    ('standard_scaler', StandardScaler())
                ])
            elif scaler == 'min_max':
                num_transformer = Pipeline(steps=[
                    ('standard_scaler', MinMaxScaler())
                ])

            pipeline_combined = ColumnTransformer(transformers=[
                ('numerical_preprocessing', num_transformer, numerical_columns)],
                remainder='drop',
                verbose=True)

            model_clone = clone(model_instance)
            model_clone.random_state = 42

            pipeline = Pipeline([("transform_inputs", pipeline_combined), ("classifier", model_clone)])

            pipeline.fit(new_X_train, y_train)

            drop_col_score = pipeline.score(new_X_train, y_train)
            feats[X_train.columns[index]] = (benchmark_score - drop_col_score) * 1000

        feats = {k: v for k, v in sorted(feats.items(), key=lambda item: item[1])}

        plt.figure(figsize=(20, 15))
        plt.bar(range(len(feats)), list(feats.values()), align='center')
        # plt.legend(handles=self.plt_legend)
        plt.axhline(color='black', lw=2, label='0')
        plt.xticks(range(len(feats)), list(feats.keys()), rotation=45, fontsize=20, horizontalalignment='right')
        plt.ylim(min(feats.values()), max(feats.values()))
        plt.ylabel('Drop Feat. Importance in %')
        plt.yticks(np.arange(min(feats.values()), max(feats.values()), step=5), fontsize=20)
        plt.tight_layout()
        plt.savefig('drop_importance_n_100.png')
        plt.show()

        print('\nBenchmark Score: ', benchmark_score)
