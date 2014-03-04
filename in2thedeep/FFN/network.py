import cPickle


#TODO add a more general class ? 
class FFNetwork():
    """ feed forward general Network """
    def __init__(self, input, list_of_layers_builder):
        self.layers = []
        temp_input = input
        for layer_builder in list_of_layers_builder:
            self.layers.append(layer_builder.get_layer(temp_input))
            temp_input = self.layers[-1].output
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)

    def save_model(self, path='model.tkl'):
        with open(path) as f:
            cPickle.dump(self.params, f)

    def get_symetric_builder(self):
        """ construct the symetric of a Network """
        return FFNetworkBuilder([layer.get_symmetric_builder() for layer in reversed(self.layers)])

    def add_layer(self, layer_builder):
        temp_input = self.layers[-1].output
        self.layers.append(layer_builder.get_layer(temp_input))
        self.params.append(self.layers[-1].params)

    def union(self, network2_builder):
        #TODO change this with a builder
        """ connect one network to another """
        for layer_builder in network2_builder.list_of_layers_builder:
            self.add_layer(layer_builder)

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


class FFNetworkBuilder():
    def __init__(self, list_of_layers_builder):
        self.list_of_layers_builder = list_of_layers_builder

    def get_network(self, input):
        return FFNetwork(input, self.list_of_layers_builder)

