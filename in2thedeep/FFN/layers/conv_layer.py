import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano
import theano.tensor as T
from in2thedeep.FFN.layer import Layer, LayerBuilder, LayerInfos


class LeNetConvPoolLayerInfos(LayerInfos):
    def __init__(self, dict=None):
        """   :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        : param pooling_on: bool to do or not the max_pooling step
       """

        if dict is None:
            raise NotImplementedError("Need at least an architecture")
        self.infos = dict
        if self.infos.get('activation') is None:
            self.infos['activation'] = T.tanh
        if self.infos.get('pooling_on') is None:
            self.infos['pooling_on'] = True
        if self.infos.get('border_mode') is None:
            self.infos['border_mode'] = "valid"  # full or valid parameter of convolution

        assert(self.infos.get('filter_shape') is not None)
        assert(self.infos.get('image_shape') is not None)
        assert(self.infos.get('poolsize') is not None)

        assert self.infos['image_shape'][1] == self.infos['filter_shape'][1]

    def complete_infos(self):
        fan_in = np.prod(self.filter_shape[1:])
                # the bias is a 1D tensor -- one bias per output feature map

        if self.infos.get('W') is None:
            print """ random init """
            assert(self.infos.get('n_in') is not None)
            assert(self.infos.get('n_out') is not None)

            W = np.asarray(self.rng.uniform(
                low=-np.sqrt(3. / fan_in),
                high=np.sqrt(3. / fan_in),
                size=self.filter_shape), dtype=theano.config.floatX)

            self.infos['W'] = W

        if self.infos.get['b'] is None:
            b = np.zeros((self.infos['filter_shape'][0],), dtype=theano.config.floatX)
            self.infos['b'] = b

        if self.infos.get('n_in') is None:
            raise NotImplementedError(' recreating architecture from W and b is not done yet')

        output_shape = list(self.infos['image_shape'])
        output_shape[1] = self.infos['filter_shape'][0]
        if self.infos.get['border_mode'] is "valid":
            output_shape[2] = self.infos['image_shape'][2] - self.infos['filter_shape'][2] + 1
            output_shape[3] = self.infos['image_shape'][3] - self.infos['filter_shape'][3] + 1
        else:
            output_shape[2] = self.infos['image_shape'][2] + self.infos['filter_shape'][2] - 1
            output_shape[3] = self.infos['image_shape'][3] + self.infos['filter_shape'][3] - 1


class LeNetConvPoolLayer(Layer):
    def __init__(self, input, layer_infos):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        :type input: theano.tensor.dtensor4
        """
        self.layer_infos = layer_infos.get_params()
        self.input = input

        self.rng = np.random.RandomState(1234)

        self.W = theano.shared(value=self.layer_infos['W'], name='W')
        self.b = theano.shared(value=self.layer_infos['b'], name='b')

        conv_out = conv.conv2d(input, self.W, border_mode=self.layer_infos['border_mode'], filter_shape=self.filter_shape, image_shape=self.image_shape)

        if self.layer_infos['pooling_on']:
            pooled_out = downsample.max_pool_2d(conv_out, self.poolsize, ignore_border=True)
        else:
            pooled_out = conv_out

        self.output = self.layer_infos['activation'](pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

    def get_symmetric_builder(self):
        layer_constructor = LeNetConvPoolLayer
        #print self.W.get_value().shape
        fshape = self.layer_infos['filter_shape']
        infos = {
            'W':  self.W.get_value().transpose(1, 0, 3, 2),
            'b': np.zeros((self.filter_shape[1],), dtype=theano.config.floatX),
            'activation': self.layer_infos['activation'],
            'pooling_on': False,
            'border_mode': 'full',
            'filter_shape': [fshape[1], fshape[0], fshape[2], fshape[3]],
            'image_shape': list(self.layer_infos['output_shape'])
        }

        assert(not self.layer_infos['pooling_on'])  # for autoencode Max pooling needs to be deactivated
        return LayerBuilder(layer_constructor, LeNetConvPoolLayerInfos(infos))

    @staticmethod
    def input_structure():
        return "tensor4"
