from in2thedeep.OM.optim_infos import OptimInfos
import numpy as np
from in2thedeep.FFN.network import NetworkTester
from in2thedeep.FFN.network import NetworkTransformer
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
        self.transform_ready = False

    def fit(self, X, Y=None, valid_set=None, test_set=None):
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
            dataset_y = X  # np.copy(X)
        else:
            print "building net ..."
            self.net = self.network_architect.build_network(x) 
            dataset_x = X
            dataset_y = Y
        
        dataset_x = theano.shared(dataset_x, borrow=True) 
        dataset_y = theano.shared(dataset_y, borrow=True) 
        if valid_set is not None:
            valid_set = list(valid_set)
            valid_set[0] = theano.shared(valid_set[0], borrow=True)
            valid_set[1] = theano.shared(valid_set[1], borrow=True)
    
        if test_set is not None:
            test_set = list(test_set)
            test_set[0] = theano.shared(test_set[0], borrow=True)
            test_set[1] = theano.shared(test_set[1], borrow=True)
    
        datasets = ((dataset_x, dataset_y), valid_set, test_set)
        self.tester = NetworkTester(self.net, y, method=self.optim_infos.infos['cost_method'], take=self.optim_infos.infos['take'])
        print "...done"
    
        print self.net
        self.optim_infos.run_method(datasets, self.tester)

    def eval(self, X, Y=None):
        """ evaluation"""
        try:
            assert(self.network_architect is not None)
        except:
            raise NotImplementedError("You need to provide parameters to build the network")

        x = T.matrix('x')
        y = T.matrix('y')
   
        if Y is None:
            raise NotImplementedError("autoencoder have to test mode.")
        else:
            print "building net ..."
            self.net = self.network_architect.build_test_net(x)
            dataset_x = X
            dataset_y = Y
        
        dataset_x = theano.shared(dataset_x, borrow=True) 
        dataset_y = theano.shared(dataset_y, borrow=True) 
   
        self.tester = NetworkTester(self.net, y)
        cost = self.tester.get_cost()
        batch_size = self.optim_infos.infos['batch_size']
        self.index = T.lscalar()    # index to a [mini]batch
        self.eval_net = theano.function(
            [self.index], cost,
            givens={
                self.tester.input: dataset_x[self.index * batch_size:(self.index + 1) * batch_size],
                self.tester.y_values: dataset_y[self.index * batch_size:(self.index + 1) * batch_size]})
        c = []

        self.n_train_batches = int(dataset_x.get_value().shape[0] / batch_size)
        for batch_index in xrange(self.n_train_batches):
            c.append(self.eval_net(batch_index))
        print 'valid', np.mean(c)
        print "...done"
        return np.mean(c)
  
    def transform(self, X):
        """ evaluation"""
        try:
            assert(self.network_architect is not None)
        except:
            raise NotImplementedError("You need to provide parameters to build the network")
        dataset_x = X
        dataset_x = theano.shared(dataset_x, borrow=True) 
        batch_size = X.shape[0]

        if not self.transform_ready:
            self.transform_ready = True
            x = T.matrix('x')
    
            print "building net ..."
            self.optim_infos.infos['batch_size'] = batch_size
    
            #self.network_architect.change_batch_size(batch_size)
            self.net = self.network_architect.build_test_net(x)
            self.tester = NetworkTransformer(self.net)
            cost = self.tester.predict()
            #batch_size = self.optim_infos.infos['batch_size']
            self.index = T.lscalar()    # index to a [mini]batch
            self.eval_net = theano.function(
                [self.index], cost,
                givens={
                    self.tester.input: dataset_x[self.index * batch_size:(self.index + 1) * batch_size]})
        c = []

        self.n_train_batches = int(dataset_x.get_value().shape[0] / batch_size)
        for batch_index in xrange(self.n_train_batches):
            c.append(self.eval_net(batch_index))
        #c = np.array(c)
        c = np.concatenate(c)
        #print c.shape
        return c
  
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, y):
        pass

    def set_params(self, network_architect):
        self.network_architect = network_architect

    def get_params(self):
        return self.network_architect
