from in2thedeep.FFN.layer import Layer, LayerInfos
import theano.tensor as T
import theano
import numpy as np


class HiddenLayerInfos(LayerInfos):
    def __init__(self, dict=None):
        """ keys of infos : W, b, activation, n_in, n_out, dropout, dropout_rate, dropout_test """
        self.rng = np.random.RandomState(1234)
        if dict is None:
            raise NotImplementedError("Need at least an architecture")
        self.infos = dict
        if self.infos.get('activation') is None:
            self.infos['activation'] = theano.tensor.nnet.sigmoid

        if self.infos.get('dropout') is None:
            self.infos['dropout'] = False

        if self.infos.get('dropout_test') is None:
            self.infos['dropout_test'] = False
        self.infos['constructor'] = HiddenLayer
        self.infos['input_structure'] = 'matrix' 

    def complete_infos(self):
        if self.infos.get('W') is None:
            print """ random init """
            assert(self.infos.get('n_in') is not None)
            assert(self.infos.get('n_out') is not None)

            n_in = self.infos['n_in']
            n_out = self.infos['n_out']
            W = np.asarray(self.rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if self.infos['activation'] == theano.tensor.nnet.sigmoid:
                W *= 4

            self.infos['W'] = W

        if self.infos.get('b') is None:
            b = np.zeros(self.infos['n_out'], dtype=theano.config.floatX)
            self.infos['b'] = b

        if self.infos.get('n_in') is None:
            raise NotImplementedError(' recreating architecture from W and b is not done yet')

    def __str__(self):
        return """ Multi Perception Layer :
            n_in : {}
            n_out : {}
            """.format(self.infos['n_in'], self.infos['n_out'])


class HiddenLayer(Layer):
    def __init__(self, input, layer_infos):
        """
            W and b are shared variables
        """
        self.layer_infos = layer_infos

        self.W = theano.shared(self.layer_infos['W'], borrow=True)
        self.b = theano.shared(self.layer_infos['b'], borrow=True)

        self.rng = np.random.RandomState(1234)
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(
            self.rng.randint(999999)) 
        if self.layer_infos['dropout_test']:
            self.W *= self.layer_infos['dropout_rate']  # mean for testing after dropout training TODO change this to make it possible during training
                     
        self.output = self.layer_infos['activation'](T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]
        
        if self.layer_infos['dropout']:
            mask = self.srng.binomial(n=1, p=1-self.layer_infos['dropout_rate'], size=self.output.shape)
            self.output = self.output * T.cast(mask, theano.config.floatX)
      
    def get_symmetric_infos(self):
        infos = {
            'n_in': self.layer_infos['n_out'],
            'n_out': self.layer_infos['n_in'],
            'activation': self.layer_infos['activation'],
            'W': self.W.get_value().T,  # we export W but not b
        }
        layer_info = HiddenLayerInfos(infos)
        return layer_info

    def get_infos(self):
        self.layer_infos['W'] = self.W.get_value()
        self.layer_infos['b'] = self.b.get_value()
        return HiddenLayerInfos(self.layer_infos)

    def __str__(self):
        return HiddenLayerInfos(self.layer_infos).__str__()
