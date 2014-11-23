import cPickle
import theano.tensor as T


#TODO add a more general class ? 
class FFNetwork():
    """ feed forward general Network, in case of not using conv net, disard params dataset_shape, input_shape """
    def __init__(self, input, list_of_layers_infos,  dataset_shape=(1, 1, 32, 32), input_type='matrix'):
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
        print "Network depth , ", len(list_of_layers_infos)
        for layer_infos in list_of_layers_infos:
            print layer_infos.infos['constructor']
            self.add_layer(layer_infos)

    def save_model(self, path='model.tkl'):
        list_of_layers_infos = [layer.get_infos() for layer in self.layers]
        with open(path, 'w') as f:
            cPickle.dump(list_of_layers_infos, f)

    def get_symmetric_infos(self):
        """ construct the symetric of a Network """
        return [layer.get_symmetric_infos() for layer in reversed(self.layers)]

    def add_layer(self, layer_infos):
        try:
            temp_input = self.layers[-1].output
        except:
            temp_input = self.input

        #potential reshaping
        next_type = layer_infos.get_input_structure()
        if self.temp_type != next_type:
            if len(self.layers) == 0:
                print " adding first layer"
                temp_input = self.reshape_mapping[(self.temp_type, next_type)](temp_input, self.dataset_shape)
            else:
                temp_input = self.reshape_mapping[(self.temp_type, next_type)](temp_input, layer_infos.get_output_shape())
            self.temp_type = next_type
        
        layer = layer_infos.get_layer(temp_input)
        self.layers.append(layer)
        for param in self.layers[-1].params:
            self.params.append(param)
        self.output = self.layers[-1].output.flatten(2)  # output always matrix !

    def union(self, network2_infos):
        """ connect one network to another """
        for layer_infos in network2_infos:
            self.add_layer(layer_infos)

    def __str__(self):
        res = "#######Architecture#######\n "
        for layer in self.layers:
            res += layer.__str__()
            res += "\n" + "           //||\\        \n"
        res += "----------------"
        return res


class NetworkTransformer():
    def __init__(self, network):
        self.network = network
        self.input = self.network.input

    def predict(self):
        return self.network.output
  

class NetworkTester():
    """ get errors function for layer """
    def __init__(self, network, y_values, path='model.tkl'):
        self.network = network
        self.y_values = y_values
        self.input = self.network.input
        self.saving_path = path

    def get_cost_updates(self, learning_rate=0.1, method="rmse"):
#TODO change this to do the grad someplace else
        cost = self.get_cost(method)
        print "cost type", method

        #print " params type", type(self.network.params[0])
        gparams = T.grad(cost, self.network.params)
        updates = []
        for param, gparam in zip(self.network.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return (cost, updates)

    def save(self):
        self.network.save_model(self.saving_path)

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
   
