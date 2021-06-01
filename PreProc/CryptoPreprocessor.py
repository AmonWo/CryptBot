import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

NUMERICAL_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']


class CryptoPreprocessor:
    def __init__(self, classifier, dataframe):
        print('CRYPTOPREPROCESSOR INITIALIZED\n')
        self.classifier = classifier
        self.pipeline = self.create_pipeline(dataframe)

    def create_pipeline(self, df):

        num_cols = df.columns[df.dtypes != 'object'].tolist()

        num_transformer = Pipeline(steps=[
            ('min_max_scaler', MinMaxScaler())
        ])
        pipeline_combined = ColumnTransformer(transformers=[
            ('numerical_preprocessing', num_transformer, num_cols)],
            remainder='drop',
            verbose=True)
        print('PIPELINE CREATED\n')
        return Pipeline([("transform_inputs", pipeline_combined), ("classifier", self.classifier)])

    def split_train_test(self, df=None, X=None, y=None):
        if X is None and y is None:
            X, y = df.drop('Prediction', axis=1), df['Prediction']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # print('\nX TRAIN COLS\n', X_train.columns)
        return X_train, X_test, y_train, y_test

    def select_date_range(self, df, start_date='2015-01-01', end_date='2020-01-01'):
        print('Select data between ', start_date, ' and ', end_date)
        train_data = pd.DataFrame(df.loc[start_date:end_date])
        # test_data = pd.DataFrame(df.loc[end_date:])
        return train_data

    @staticmethod
    def sample_label(df=None, upsampling=False, multiplier=1):
        """
        Samples given dataset to equally distributed labels
        :param df: Dataframe to be sampled
        :param upsampling: Upsampling (True), Downsampling (False)
        :param multiplier: Multiplier to increase the size of the dataframe by randomly picking data
        :return: Sampled dataframe
        """
        print('\nPRE SAMPLING SHAPE\n', df.shape)
        sell = df[df.Prediction < -.5]
        buy = df[df.Prediction > .5]
        print('\nPRE SAMPLING COMBINED SELL/BUY\n', len(sell), len(buy))

        if upsampling:
            # Upsampling
            y_upsampled = buy.sample(n=len(sell.Prediction), replace=True, random_state=42)
            df = y_upsampled.append(sell)
            method = 'UPSAMPLING'
        else:
            # Downsampling
            y_churned = buy.sample(n=(len(buy.Prediction) * multiplier), replace=True,
                                   random_state=42)
            y_downsampled = sell.sample(n=(len(buy.Prediction) * multiplier), replace=True,
                                        random_state=42)
            df = y_downsampled.append(y_churned)
            method = 'DOWNSAMPLING'

        sell = df[df.Prediction < -.5]
        buy = df[df.Prediction > .5]
        print('\nPOST {} COMBINED SELL/BUY\n'.format(method), len(sell), len(buy))

        return df

    @staticmethod
    def remove_nans(df):
        pre_drop_na = len(df)
        df.dropna(axis=0, inplace=True)
        post_drop_na = len(df)
        print('LOST ROWS BECAUSE OF NAN: ', pre_drop_na - post_drop_na, "\n")
        return df
