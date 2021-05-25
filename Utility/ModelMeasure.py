import matplotlib.pyplot as plt

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
