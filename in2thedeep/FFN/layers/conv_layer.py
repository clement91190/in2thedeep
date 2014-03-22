import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano
import theano.tensor as T
from in2thedeep.FFN.layer import Layer, LayerInfos


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
        self.rng = np.random.RandomState(1234)
        if dict is None:
            raise NotImplementedError("Need at least an architecture")
        self.infos = dict
        if self.infos.get('activation') is None:
            self.infos['activation'] = T.tanh
        if self.infos.get('pooling_on') is None:
            self.infos['pooling_on'] = True
        if self.infos.get('border_mode') is None:
            self.infos['border_mode'] = "valid"  # full or valid parameter of convolution
        if self.infos.get('ignore_border') is None:
            self.infos['ignore_border'] = True  # same for maxpooling op
        if self.infos.get('poolsize') is None:
            self.infos['poolsize'] = (2, 2)  # 

        self.infos['constructor'] = LeNetConvPoolLayer
        self.infos['input_structure'] = 'tensor4'
        assert(self.infos.get('filter_shape') is not None)
        assert(self.infos.get('image_shape') is not None)
        #assert(self.infos.get('poolsize') is not None)

        assert self.infos['image_shape'][1] == self.infos['filter_shape'][1]

    def change_batch_size(self, batch_size):
        shape = self.infos['image_shape']
        shape = list(shape)
        shape[0] = batch_size
        self.infos['image_shape'] = tuple(shape)

    def complete_infos(self):
        fan_in = np.prod(self.infos['filter_shape'][1:])
                # the bias is a 1D tensor -- one bias per output feature map

        if self.infos.get('W') is None:
            print """ random init """

            W = np.asarray(self.rng.uniform(
                low=-np.sqrt(3. / fan_in),
                high=np.sqrt(3. / fan_in),
                size=self.infos['filter_shape']), dtype=theano.config.floatX)

            self.infos['W'] = W

        if self.infos.get('b') is None:
            b = np.zeros((self.infos['filter_shape'][0],), dtype=theano.config.floatX)
            self.infos['b'] = b

        output_shape = list(self.infos['image_shape'])
        output_shape[1] = self.infos['filter_shape'][0]
        if self.infos['border_mode'] is "valid":
            output_shape[2] = self.infos['image_shape'][2] - self.infos['filter_shape'][2] + 1
            output_shape[3] = self.infos['image_shape'][3] - self.infos['filter_shape'][3] + 1
        else:
            output_shape[2] = self.infos['image_shape'][2] + self.infos['filter_shape'][2] - 1
            output_shape[3] = self.infos['image_shape'][3] + self.infos['filter_shape'][3] - 1

        if self.infos['pooling_on']:
            print "careful with output shape ?"
            output_shape[2] /= self.infos['poolsize'][0]
            output_shape[3] /= self.infos['poolsize'][0]
            if not self.infos['ignore_border']:
                output_shape[2] += 1
                output_shape[3] += 1
        self.infos['output_shape'] = output_shape

    def get_output_shape(self):
        return self.infos['output_shape']


class LeNetConvPoolLayer(Layer):
    def __init__(self, input, layer_infos):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        :type input: theano.tensor.dtensor4
        """
        self.layer_infos = layer_infos
        #print self.layer_infos
        self.input = input

        self.W = theano.shared(value=self.layer_infos['W'], name='W')
        self.b = theano.shared(value=self.layer_infos['b'], name='b')

        conv_out = conv.conv2d(
            input, self.W,
            border_mode=self.layer_infos['border_mode'],
            filter_shape=self.layer_infos['filter_shape'],
            image_shape=self.layer_infos['image_shape'])

        if self.layer_infos['pooling_on']:
            pooled_out = downsample.max_pool_2d(conv_out, self.layer_infos['poolsize'], self.layer_infos['ignore_border'])
        else:
            pooled_out = conv_out

        self.output = self.layer_infos['activation'](pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

    def get_symmetric_infos(self):
        fshape = self.layer_infos['filter_shape']
        infos = {
            'W':  self.W.get_value().transpose(1, 0, 3, 2),
            'b': np.zeros((self.layer_infos['filter_shape'][1],), dtype=theano.config.floatX),
            'activation': self.layer_infos['activation'],
            'pooling_on': False,
            'border_mode': 'full',
            'filter_shape': [fshape[1], fshape[0], fshape[2], fshape[3]],
            'image_shape': list(self.layer_infos['output_shape'])
        }

        assert(not self.layer_infos['pooling_on'])  # for autoencode Max pooling needs to be deactivated
        return LeNetConvPoolLayerInfos(infos)

    def get_infos(self):
        self.layer_infos['W'] = self.W.get_value()
        self.layer_infos['b'] = self.b.get_value()
        return LeNetConvPoolLayerInfos(self.layer_infos)
