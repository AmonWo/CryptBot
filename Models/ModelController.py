import numpy as np


class ModelController:
    def predict(self, test_data, preproc):
        """
        Appends prediction to the given dataset. Predicts buy(1) sell(1) recommendations
        :param test_data:
        :param preproc:
        :return:
        """
        test_data['Prediction'] = np.around(preproc.pipeline.predict(test_data), 0).astype(int)
        return test_data
