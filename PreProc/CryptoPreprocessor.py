from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

NUMERICAL_COLUMNS = ['Open', 'High', 'Low', 'Volume', 'Trades']


class CryptoPreprocessor:
    def __init__(self, classifier):
        print('CryptoPreproc initialized')
        # self.dataframe = dataframe
        self.classifier = classifier
        self.pipeline = self.create_pipeline()

    def create_pipeline(self):
        num_transformer = Pipeline(steps=[
            ('min_max_scaler', MinMaxScaler())
        ])
        pipeline_combined = ColumnTransformer(transformers=[
            ('numerical_preprocessing', num_transformer, NUMERICAL_COLUMNS)],
            remainder='drop',
            verbose=True)
        print('PIPELINE CREATED')
        return Pipeline([("transform_inputs", pipeline_combined), ("classifier", self.classifier)])

    def split_train_test(self, df=None, X=None, y=None):
        if X is None and y is None:
            X, y = df.drop('Prediction', axis=1), df['Prediction'].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # print('\nX TRAIN COLS\n', X_train.columns)
        return X_train, X_test, y_train, y_test

    def prepare(self, df):
        print('PREPARE')
