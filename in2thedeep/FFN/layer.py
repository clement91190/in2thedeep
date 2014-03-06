""" general type for representing a layer of a network """
from exceptions import NotImplementedError


class Layer():
    def __init__(self, input, architecture, params=None):
        self.input = input
        if params is None:
            if architecture is None:
                raise(Exception('no parameters nor architecture given'))
            self.architecture = architecture    
            self.params = self.random_init(architecture)
        else:
            self.params = params
            #self.architecture = self.get_architecture_from_params()

    def random_init(self, architecture):
        raise NotImplementedError('random initialization')

    def get_symmetric_builder(self):
        raise NotImplementedError('construct symmetric layer')

    def __str__(self):
        return "{}".format(self.architecture)

    @staticmethod
    def input_structure():
        """ describe the shape of the input 
         -> "matrix" (classif(batch_size, feature_size))
         -> "tensor4" for LENETMaxPoolLAyer(batch_size, feature_map _size(RGB channels, image_width, image_height)))"""
        raise NotImplementedError('must precise the shape of input')


class LayerBuilder():
    def __init__(self, layer_constructor, architecture, params):
        self.architecture = architecture
        self.params = params
        self.layer_constructor = layer_constructor

    def get_layer(self, input):
        return self.layer_constructor(input, self.architecture, self.params)
    
    def input_structure(self):
        return self.layer_constructor.input_structure()


