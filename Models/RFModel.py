import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier


class RFModel:
    """
    ModelClasss
    """

    def __init__(self):
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
                                            max_samples=50,
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
