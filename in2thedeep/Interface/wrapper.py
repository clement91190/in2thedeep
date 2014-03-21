from in2thedeep.OM.optim_infos import OptimInfos
import numpy as np
from in2thedeep.FFN.network import NetworkTester
import theano.tensor as T
import theano


class Wrapper():
    """ class to use the sklearn syntax """
    def __init__(self, network_architect, optim_infos=OptimInfos()):
        """ network_architect -> object of type NetworkArchitect
            optim_infos : information for the Learning method """
#TODO add default option ? 
        self.network_architect = network_architect
        self.optim_infos = optim_infos

    def fit(self, X, Y=None):
        """ train the algorithm """
        try:
            assert(self.network_architect is not None)
        except:
            raise NotImplementedError("You need to provide parameters to build the network")

        x = T.matrix('x')
        y = T.matrix('y')
   
        if Y is None:
            print "no y given -> training autoencoder"
            self.net = self.network_architect.build_autoencoder(x) 
            dataset_x = X
            dataset_y = X #np.copy(X)
        else:
            print "building net ..."
            self.net = self.network_architect.build_network(x) 
            dataset_x = X
            dataset_y = Y
        
        dataset_x = theano.shared(dataset_x) 
        dataset_y = theano.shared(dataset_y) 
    
        datasets = (dataset_x, dataset_y)
        self.tester = NetworkTester(self.net, y)
        print "...done"
    
        print self.net
        self.optim_infos.run_method(datasets, self.tester)

    def transform(self, X):
        """ transform to feature space """
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, y):
        pass

    def set_params(self, network_architect):
        self.network_architect = network_architect

    def get_params(self):
        return self.network_architect
