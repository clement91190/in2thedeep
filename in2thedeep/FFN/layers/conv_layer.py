import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano
import theano.tensor as T
from in2thedeep.FFN.layer import Layer, LayerBuilder


class LeNetConvPoolLayer(Layer):

    def __init__(self, input, (filter_shape, image_shape, poolsize), (W_values, b_values)=(None, None), (pooling_on, encode)= (True,False)):
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
        : param pooling_on: bool to do or not the max_pooling step 
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        self.rng = np.random.RandomState(1234)
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.architecture = (filter_shape, image_shape, poolsize)
        self.pooling_on=pooling_on
        


        
        if W_values is None:
            W_values, b_values = self.random_init()
        
        self.W = theano.shared(value=W_values, name='W')
        self.b = theano.shared(value=b_values, name='b')
        
        if encode:
            self.output_shape = list(self.image_shape)  # [self.image_shape[0], self.image_shape[1], (self.image_shape[2]   
            self.output_shape[1] = self.filter_shape[0]
            self.output_shape[2] = self.image_shape[2] - self.filter_shape[2] + 1 
            self.output_shape[3] = self.image_shape[3] - self.filter_shape[3] + 1 

            conv_out = conv.conv2d(input, self.W, border_mode='valid', filter_shape=self.filter_shape, image_shape=self.image_shape)
        else:
            conv_out = conv.conv2d(input, self.W, border_mode='full', filter_shape=self.filter_shape, image_shape=self.image_shape)
            self.output_shape = list(self.image_shape)  # [self.image_shape[0], self.image_shape[1], (self.image_shape[2]   
            self.output_shape[1] = self.filter_shape[0]
            self.output_shape[2] = self.image_shape[2] + self.filter_shape[2] - 1 
            self.output_shape[3] = self.image_shape[3] + self.filter_shape[3] - 1 

        if pooling_on:
            pooled_out = downsample.max_pool_2d(conv_out, self.poolsize, ignore_border=True)
            #self.output_shape[2] = self.image_shape[2] + self.filter_shape[2] - 1 
            #self.output_shape = self.output_shape  
            #TODO change the shape when maxpool on
        else:
            pooled_out = conv_out
       
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.W, self.b]
        
    def random_init(self):
        fan_in = np.prod(self.filter_shape[1:])
        W_values = np.asarray(self.rng.uniform(
            low=-np.sqrt(3./fan_in),
            high=np.sqrt(3./fan_in),
            size=self.filter_shape), dtype=theano.config.floatX)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        return (W_values, b_values)

    def get_symmetric_builder(self):
        layer_constructor = LeNetConvPoolLayer
        print self.W.get_value().shape
        W_ = self.W.get_value().transpose(1, 0, 3, 2)
        print W_.shape
        b_ = np.zeros((self.filter_shape[1],), dtype=theano.config.floatX)
        mode = (False, False)  # TODO write this better
        filter_shape = self.filter_shape
        filter_shape =[filter_shape[1], filter_shape[0], filter_shape[2], filter_shape[3]]
        image_shape = self.output_shape
        assert image_shape[1] == filter_shape[1]
        pool_size = (1, 1)
        params = [W_, b_]
        assert(not self.pooling_on)  # for autoencode Max pooling needs to be deactivated
        return LayerBuilder(layer_constructor, (filter_shape, image_shape, pool_size), params, mode)

    @staticmethod
    def input_structure():
        return "tensor4"

