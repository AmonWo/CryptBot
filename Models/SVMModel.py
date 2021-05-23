from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier



class SVMModel:
    """
    ModelClasss
    """

    def __init__(self, transformer, train_data, X_train, X_test, y_train, y_test):
        self.transformer = transformer
        self.train_data = train_data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_trans = self.transformer.fit_transform(X_train)
        self.X_test_trans = self.transformer.fit_transform(X_test)
        self.pred_train = None
        self.pred_test = None
        self.plt_legend = []
        self.model = self.create_model()
        self.features = ['Firma', 'Privat', 'Newsletter_N', 'Newsletter_Y', 'AMEX', 'Lastschrift', 'Mastercard',
                         'Keine Zahlungsart', 'Paypal', 'Visacard', 'Tage_seit_Anmeldung', 'Tage_seit_letzter_Änderung',
                         'Gesamtanzahl Bestellungen', 'Gesamtumsatz', 'Tage_seit_letzter_Bestellung']

    def create_model(self):
        # clf_svm = OneVsRestClassifier(SVC(verbose=True, cache_size=1000), n_jobs=-1)

        clf_svm = SVC(kernel='rbf', verbose=True, cache_size=1000, gamma='auto', probability=False)
        clf_svm.fit(self.X_train_trans, self.y_train)

        self.pred_train = clf_svm.predict(self.X_train_trans)
        self.pred_test = clf_svm.predict(self.X_test_trans)

        return clf_svm

    def print_accuracy(self):
        print(
            '\nMean Test Accuracy: {:.2f}\nPrediction Train Accuracy Score: {:.2f} \nPrediction Test Accuracy Score: {:.2f}'.format(
                self.model.score(self.X_test_trans, self.y_test),
                accuracy_score(self.y_train, self.pred_train),
                accuracy_score(self.y_test, self.pred_test)
            ))

    def print_report(self):
        print(classification_report(self.y_test, self.pred_test))
