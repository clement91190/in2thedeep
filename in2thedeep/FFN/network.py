import cPickle


#TODO add a more general class ? 
class FFNetwork():
    """ feed forward general Network """
    def __init__(self, list_of_layers):
        self.list_of_layers = list_of_layers  # see layer type
        self.params = []
        for layer in self.list_of_layers:
            self.params.append(layer.params)

    def save_model(self, path='model.tkl'):
        with open(path) as f:
            cPickle.dump(self.params, f)

    def get_symetric(self):
        """ construct the symetric of a Network """
        return FFNetwork([layer.get_symmetric() for layer in reversed(self.list_of_layers)])

    def add_layer(self,layer):
        self.list_of_layers.append(layer)
        self.params.append(layer.params)

    def union(self, network2):
        """ connect one network to another """
        for layer in network2.list_of_layers:
            self.add_layer(layer)

    def __str__(self):
        res = "#######Architecture#######"
        for layer in self.list_of_layers:
            res += layer.__str__()
            res += "\n" + "           //||\\        \n"
        res += "----------------"


def Autoencoder(Network):
    """ construct a Autoencoder from any Network """
    def __init__(self, network):
        Network.__init__(self, network.list_of_layers)
        self.union(self.get_symmetric())
    
