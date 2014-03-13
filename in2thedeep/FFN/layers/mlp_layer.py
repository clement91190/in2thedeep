from in2thedeep.FFN.layer import Layer, LayerBuilder, LayerInfos
import theano.tensor as T
import theano
import numpy as np


class HiddenLayerInfos(LayerInfos):
    def __init__(self, dict=None):
        if dict is None:
            raise NotImplementedError("Need at least an architecture")
        self.infos = dict
        if self.infos.get('activation') is None:
            self.infos['activation'] = theano.tensor.nnet.sigmoid

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
            if self.activation == theano.tensor.nnet.sigmoid:
                W *= 4

            self.infos['W'] = W

        if self.infos.get['b'] is None:
            b = np.zeros(self.infos['n_out'], dtype=theano.config.floatX)
            self.infos['b'] = b

        if self.infos.get('n_in') is None:
            raise NotImplementedError(' recreating architecture from W and b is not done yet')


class HiddenLayer(Layer):
    def __init__(self, input, layer_infos):
        """
            W and b are shared variables
        """
        self.layer_infos = layer_infos.get_params()
        self.rng = np.random.RandomState(1234)

        self.W = theano.shared(self.layer_infos['W'], borrow=True)
        self.b = theano.shared(self.layer_infos['b'], borrow=True)

        self.output = self.activation(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]

    def get_symmetric_builder(self):
        layer_constructor = HiddenLayer
        infos = {
            'n_in': self.layer_infos['n_out'],
            'n_out': self.layer_infos['n_in'],
            'activation': self.layer_infos['activation'],
            'W': self.W.get_value().T,  # we export W but not b
        }
        layer_info = HiddenLayerInfos(infos)
        return LayerBuilder(layer_constructor, layer_info)

    @staticmethod
    def input_structure():
#TODO add this into the layer infos
        return "matrix"
