import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano
import theano.tensor as T
from in2thedeep.FFN.layer import Layer


class LeNetConvPoolLayer(Layer):

    def __init__(self, input, (filter_shape, image_shape, poolsize), (W_values, b_values)=(None, None)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type input: theano.tensor.dtensor4

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        self.rng = np.random.RandomState(1234)
        self.filter_shape = filter_shape
        
        if W_values is None:
            W_values, b_values = self.random_init()
        
        self.W = theano.shared(value=W_values, name='W')
        self.b = theano.shared(value=b_values, name='b')

        conv_out = conv.conv2d(input, self.W, filter_shape=filter_shape, image_shape=image_shape)

        pooled_out = downsample.max_pool_2d(conv_out, poolsize, ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.W, self.b]
        
        def random_init(self):
            fan_in = np.prod(self.filter_shape[1:])
            W_values = np.asarray(self.rng.uniform(
                low=-np.sqrt(3./fan_in),
                high=np.sqrt(3./fan_in),
                size=filter_shape), dtype=theano.config.floatX)

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            return (W_values, b_values)

        def get_symmetric_builder(self):
            raise NotImplementedError(" implement from stacked convolutional autoencoder Paper ?")

        @staticmethod
        def input_structure():
            return "tensor4"

