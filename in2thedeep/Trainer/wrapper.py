class Wrapper():
    """ class to use the sklearn syntax """
    def fit(self, X):
        """ train the algorithm """
        pass

    def transform(self, X):
        """ transform to feature space """
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, y):
        pass

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params
