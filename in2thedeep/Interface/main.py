from in2thedeep.Interface.network_architect import NetworkArchitect 
import numpy as np
from in2thedeep.Interface.wrapper import Wrapper


def train():
    print "building data"
    data = np.zeros((60, 1, 16, 16), dtype="float32")
    data_shape = data.shape
    #x = T.matrix('x')  # the data is presented as rasterized images
    #y = T.matrix('y')  # the data is presented as rasterized images
    
    # create a bunch of images with circle in the middle
    for im in data:
        for i in range(5):
            for j in range(5):
                if (i - 2) ** 2 + (j - 5) ** 2 < 25:
                    im[0, i, j] = 1.0
    data = data.reshape(data_shape[0], np.prod(data_shape[1:]))  # reshape in a matrix
    print "done"
    print data.shape
    
    batch_size = 20
    image_shape = list(data_shape)
    image_shape[0] = batch_size
    image_shape = tuple(image_shape)
    
    infos = {
        #'W':  self.W.get_value().transpose(1, 0, 3, 2),
        #'b': np.zeros((self.filter_shape[1],), dtype=theano.config.floatX),
        #'activation': self.layer_infos['activation'],
        'pooling_on': False,
        'border_mode': 'valid',
        'filter_shape':  (2, 1, 8, 8),
        'image_shape': image_shape
    }

    network_architect = NetworkArchitect()
    network_architect.add_conv_pool_layer(infos)
    net_trainer = Wrapper(network_architect)
    print data
    raw_input()
    net_trainer.fit(data)


def show_weight():
    network_architect = NetworkArchitect()
    network_architect.load_network()
    network_architect.from_autoencoder()
    net = network_architect.build_weight_viewer()
    print net


if __name__ == "__main__":
    train()
