from sklearn.ensemble import GradientBoostingRegressor


class GdbRegModel:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        gdb_reg = GradientBoostingRegressor(
            verbose=True,
            n_estimators=10,
            learning_rate=0.1,
            min_samples_split=5,
            max_depth=4,
            random_state=0,
            loss='ls',
        )

        return gdb_reg
