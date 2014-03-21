""" general type for representing a layer of a network """
from exceptions import NotImplementedError


class Layer():
    def __init__(self, input, layer_infos):
        self.input = input
        self.params = layer_infos.get_params()
    
    def get_symmetric_infos(self):
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

    def get_infos(self):
        self.complete_infos()
        return self.infos
    
    def change_batch_size(batch_size):
        pass

    def get_layer(self, input):
        return self.infos['constructor'](input, self.get_infos())

    def __str__(self):
        raise NotImplementedError("no string")

    def get_input_structure(self):
        return self.infos['input_structure']


