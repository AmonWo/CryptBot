from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


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
            max_depth=3,
            random_state=42,
        )

        return clf_gdb
