""" general type for representing a layer of a network """
from exceptions import NotImplementedError


class Layer():
    def __init__(self, input, layer_infos):
        self.input = input
        self.params = layer_infos.get_params()
    
    def get_symmetric_builder(self):
        raise NotImplementedError('construct symmetric layer')

    def __str__(self):
        return self.params.__str__()

    def get_infos(self):
        """ for saving the model """
        raise NotImplementedError(" return the information to build a layer (for saving)")

    @staticmethod
    def input_structure():
        """ describe the shape of the input 
         -> "matrix" (classif(batch_size, feature_size))
         -> "tensor4" for LENETMaxPoolLAyer(batch_size, feature_map _size(RGB channels, image_width, image_height)))"""
        raise NotImplementedError('must precise the shape of input')


class LayerBuilder():
    def __init__(self, layer_infos):
        self.layer_constructor = layer_infos.infos['constructor']
        print "Layer :", self.layer_constructor
        self.layer_infos = layer_infos

    def get_layer(self, input):
        return self.layer_constructor(input, self.layer_infos)
    
    def input_structure(self):
        print self.layer_constructor
        return self.layer_constructor.input_structure()


class LayerInfos():
    """class to inherit from when writing a new kind of layer,
    -> add assert statements to the parameters and default value 
    to d. """
    def init(self, dict={}):
        self.infos = dict

    def add_param(self, name, value):
        self.infos[name] = value

    def complete_infos(self):
        raise NotImplementedError('Check the information you get to the layer!')

    def get_params(self):
        self.complete_infos()
        return self.infos

    def __str__(self):
        raise NotImplementedError("no string")

