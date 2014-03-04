import cPickle
import theano.tensor as T


#TODO add a more general class ? 
class FFNetwork():
    """ feed forward general Network """
    def __init__(self, input, list_of_layers_builder):
        self.layers = []
        self.input = input
        temp_input = input
        for layer_builder in list_of_layers_builder:
            self.layers.append(layer_builder.get_layer(temp_input))
            temp_input = self.layers[-1].output
        self.output = temp_input
        self.params = []
        for layer in self.layers:
            for param in layer.params:
                self.params.append(param)

    def save_model(self, path='model.tkl'):
        with open(path) as f:
            cPickle.dump(self.params, f)

    def get_symmetric_builder(self):
        """ construct the symetric of a Network """
        return FFNetworkBuilder([layer.get_symmetric_builder() for layer in reversed(self.layers)])

    def add_layer(self, layer_builder):
        temp_input = self.layers[-1].output
        self.layers.append(layer_builder.get_layer(temp_input))
        for param in self.layers[-1].params:
            self.params.append(param)
        self.output = self.layers[-1].output

    def union(self, network2_builder):
        """ connect one network to another """
        for layer_builder in network2_builder.list_of_layers_builder:
            self.add_layer(layer_builder)

    def __str__(self):
        res = "#######Architecture#######"
        for layer in self.layers:
            res += layer.__str__()
            res += "\n" + "           //||\\        \n"
        res += "----------------"
        return res


class NetworkTester():
    """ get errors function for layer """
    def __init__(self, network, y_values):
        self.network = network
        self.y_values = y_values
        self.input = self.network.input

    def get_cost_updates(self, learning_rate=0.1, method="cross"):
#TODO change this to do the grad someplace else
        cost = self.get_cost(method)
        print "cost type", type(cost)

        print " params type", type(self.network.params[0])
        gparams = T.grad(cost, self.network.params)
        updates = []
        for param, gparam in zip(self.network.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return (cost, updates)

    def get_cost(self, method="cross"):
        y_pred = self.network.output
        if method == "cross":
            L = -T.sum(self.y_values * T.log(y_pred) + (1.0 - self.y_values) * T.log(1 - y_pred), axis=1)
            cost = T.mean(L)
        else:
            raise NotImplementedError("method not implemented -> use cross entropy error")
        return cost

    def predict(self):
        return self.network.output
    

class FFNetworkBuilder():
    def __init__(self, list_of_layers_builder):
        self.list_of_layers_builder = list_of_layers_builder

    def get_network(self, input):
        return FFNetwork(input, self.list_of_layers_builder)


class AutoEncoderBuilder(FFNetworkBuilder):
    """ construct a Autoencoder from any Network """
    def get_network(self, input):
        net = FFNetwork(input, self.list_of_layers_builder)
        net.union(net.get_symmetric_builder())
        return net

