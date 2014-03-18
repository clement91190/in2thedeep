from in2thedeep.FFN.network import FFNetwork
from in2thedeep.FFN.layers.conv_layer import LeNetConvPoolLayerInfos
import pickle


class NetworkArchitect():
    """ class to build the networks, and configure the Optimization method """
#TODO make it easier to add layers building the LayerInfos parameters
    def __init__(self, list_of_layers_infos=[]):
        self.list_of_layers_infos = list_of_layers_infos
        self.dataset_shape = (1, 1, 32, 32)
        self.input_type = 'matrix'
        self.id_of_deepest_layer = -1

    def add_layer(self, layer_info):
        """ add the layer in the list """
        self.list_of_layers_infos.append(layer_info)

    def add_conv_pool_layer(self, infos):
        self.list_of_layers_infos.append(LeNetConvPoolLayerInfos(infos))

    def set_dataset_shape(self, dataset_shape, input_type):
        """ for convolutional network, the shape of the data set need to be known """
        self.dataset_shape = dataset_shape
        self.input_type = input_type

    def check_architecture(self):
#TODO perform tests on the architecture to make sure it is possible
        pass

    def load_network(self, path="model.tkl"):
        with open(self.path, 'r') as fich:
            self.list_of_layers_infos = pickle.load(fich)

    def from_autoencoder(self):
        """ keep only first half of the layer to build a net from an autoencoder """
        self.list_of_layers_infos = self.list_of_layers_infos[:(len(self.list_of_layers_infos) / 2)]

    def build_network(self, input):
        return FFNetwork(input, self.list_of_layers_infos, self.dataset_shape, self.input_type)

    def build_autoencoder(self, input):
        """ create the autoencoder with layers[:deepest_layer] and their symmetric """
        net = FFNetwork(input, self.list_of_layers_infos[:self.id_of_deepest_layer], self.dataset_shape, self.input_type)
        net.union(net.get_symmetric_infos())
        return net

    def build_weight_viewer(self, input):
        "show the weight of the layer"
        assert(self.input_type == 'matrix')
        net = FFNetwork(input, self.list_of_layers_infos[:self.id_of_deepest_layer], self.dataset_shape, self.input_type)
        infos = net.get_symmetric_infos()
        self.dataset_shape = infos[0]['image_shape']
        assert(infos[0]['input_structure'] == 'tensor4')  # TODO adapt to see weights of MLP
        net = FFNetwork(input, infos, infos[0]['image_shape'], infos[0]['input_structure'])
        return net

    def __str__(self):
        res = "#######-- Architecture --#######\n "
        for layer in self.layers:
            res += layer.__str__()
            res += "\n" + "           //||\\        \n"
        res += "----------------"
        res += "input type", self.input_type
        res += "dataset_shape", self.dataset_shape
        return res


