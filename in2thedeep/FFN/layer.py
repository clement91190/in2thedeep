""" general type for representing a layer of a network """
from exceptions import NotImplementedError


class Layer():
    def __init__(self, architecture, params=None):
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

    def get_symmetric(self):
        raise NotImplementedError('construct symmetric layer')

