class Wrapper():
    """ class to use the sklearn syntax """
    def __init__(self, params=None):
        self.params = params
    
    def fit(self, X, y=None):
        """ train the algorithm """
        try:
            assert(self.params is not None)
        except:
            raise NotImplementedError("You need to provide parameters to build the network")
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
