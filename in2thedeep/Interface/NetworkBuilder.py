from in2thedeep.FFN.network import FFNetwork


class NetworkArchitect():
    """ class to build the networks """
#TODO make it easier to add layers building the LayerInfos parameters
    def __init__(self, list_of_layers_infos=[]):
        self.list_of_layers_infos = list_of_layers_infos
        self.dataset_shape = (1, 1, 32, 32)
        self.input_type = 'matrix'

    def add_layer(self, layer_info):
        """ add the layer in the list """
        self.list_of_layers_infos.append(layer_info)

    def set_dataset_shape(self, dataset_shape, input_type):
        """ for convolutional network, the shape of the data set need to be known """
        self.dataset_shape = dataset_shape

    def check_architecture(self):
#TODO perform tests on the architecture to make sure it is possible
        pass

    def build_network(self, input):
        return FFNetwork(input, self.list_of_layers_infos, self.dataset_shape, self.input_type)

    def build_autoencoder(self, input, id_of_deepest_layer=-1):
        """ create the autoencoder with layers[:deepest_layer] and their symmetric """
        net = FFNetwork(input, self.list_of_layers_infos[:id_of_deepest_layer], self.dataset_shape, self.input_type)
        net.union(net.get_symmetric_infos())
        return net
