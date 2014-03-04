from in2thedeep.FFN.layer import Layer, LayerBuilder
import theano.tensor as T
import theano
import numpy as np


class HiddenLayer(Layer):
    def __init__(self, input, (n_in, n_out, activation), (W, b)=(None, None)):
        """ 
            W and b are shared variables
        """
        self.rng = np.random.RandomState(1234)
        self.activation = activation
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        if W is None:
            (W, b) = self.random_init((n_in, n_out))
        
        self.W = W  
        self.b = b

        self.output = activation(T.dot(input, self.W) + T.dot(T.ones_like(T.eye(input.shape[0], 1)), self.b))
        # parameters of the model
        self.params = [self.W, self.b]

    def random_init(self, (n_in, n_out)):
        W = np.asarray(self.rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)
        if self.activation == theano.tensor.nnet.sigmoid:
            W *= 4

        #W = theano.shared(value=W_values, name='W', borrow=True)

        b = np.zeros((1, n_out), dtype=theano.config.floatX)
        #b = theano.shared(value=b_values, name='b', borrow=True)
        W = theano.shared(W, borrow=True)
        b = theano.shared(b, borrow=True)
        return (W, b)

    def get_symmetric_builder(self):
        layer_constructor = HiddenLayer
        architecture = (self.n_out, self.n_in, self.activation)
        W_ = self.W.T
        b_ = np.dot(self.W, self.b.T).T
        params = [W_, b_] 
        return LayerBuilder(layer_constructor, architecture, params)


