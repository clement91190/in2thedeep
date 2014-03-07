import cPickle
import theano.tensor as T


#TODO add a more general class ? 
class FFNetwork():
    """ feed forward general Network, in case of not using conv net, disard params dataset_shape, input_shape """
    def __init__(self, input, list_of_layers_builder,  dataset_shape=(1, 1, 32, 32), input_type='matrix'):
        self.layers = []
        self.input = input
        self.params = []
        self.output = input.flatten(2)
        self.dataset_shape = dataset_shape
        self.temp_type = input_type  
        
        self.reshape_mapping = {
            ("matrix", "tensor4"): lambda input, shape: input.reshape(shape),
            ("tensor4", "matrix"): lambda input, shape: input.flatten(2)
        }

        for layer_builder in list_of_layers_builder:
            print layer_builder.layer_constructor
            self.add_layer(layer_builder)

    def save_model(self, path='model.tkl'):
        with open(path) as f:
            cPickle.dump(self.params, f)

    def get_symmetric_builder(self):
        """ construct the symetric of a Network """
        return FFNetworkBuilder([layer.get_symmetric_builder() for layer in reversed(self.layers)])

    def add_layer(self, layer_builder):
        try:
            temp_input = self.layers[-1].output
        except:
            temp_input = self.input

        #potential reshaping
        next_type = layer_builder.input_structure()
        if self.temp_type != next_type :
            temp_input = self.reshape_mapping[(self.temp_type, next_type)](temp_input, self.dataset_shape)
            self.temp_type = next_type
        
        layer = layer_builder.get_layer(temp_input)
        self.layers.append(layer)
        for param in self.layers[-1].params:
            self.params.append(param)
        self.output = self.layers[-1].output.flatten(2)  #output always matrix !

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

    def get_cost_updates(self, learning_rate=0.1, method="rmse"):
#TODO change this to do the grad someplace else
        cost = self.get_cost(method)
        print "cost type", type(cost)

        print " params type", type(self.network.params[0])
        gparams = T.grad(cost, self.network.params)
        updates = []
        for param, gparam in zip(self.network.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return (cost, updates)

    def get_cost(self, method="rmse"):
        y_pred = self.network.output
        if method == "cross":
            L = -T.sum(self.y_values * T.log(y_pred) + (1.0 - self.y_values) * T.log(1 - y_pred), axis=1)
            cost = T.mean(L)
        elif method == "rmse":
            print "rmse"
            print self.y_values
            L = T.sqrt(T.mean(T.square(y_pred - self.y_values), axis=1))
            cost = T.mean(L)
        else:
            raise NotImplementedError("method not implemented -> use cross entropy error")
        return cost

    def predict(self):
        return self.network.output
    

class FFNetworkBuilder():
    def __init__(self, list_of_layers_builder):
        self.list_of_layers_builder = list_of_layers_builder

    def get_network(self, input, dataset_shape=(1, 1, 32, 32), input_type='matrix'):
        return FFNetwork(input, self.list_of_layers_builder, dataset_shape, input_type)


class AutoEncoderBuilder(FFNetworkBuilder):
    """ construct a Autoencoder from any Network """
    def get_network(self, input, dataset_shape=(1, 1, 32, 32), input_type='matrix'):
        net = FFNetwork(input, self.list_of_layers_builder, dataset_shape, input_type)
        net.union(net.get_symmetric_builder())
        return net

