from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.patches as mpatches


class GDBModel:
    """
    ModelClasss
    """

    def __init__(self):
        self.plt_legend = []
        self.model = self.create_model()

    def create_model(self):
        clf_gdb = GradientBoostingClassifier(
            verbose=True,
            warm_start=False,
            n_estimators=100,
            learning_rate=0.01,
            max_depth=5,
            random_state=42,
            n_iter_no_change=10
        )
        params = clf_gdb.get_params()
        for key in params:
            label = key + ': ' + str(params[key])
            self.plt_legend.append(mpatches.Patch(color='red', label=label))
        return clf_gdb
