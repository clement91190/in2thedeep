from in2thedeep.Interface.network_architect import NetworkArchitect 
import matplotlib.pyplot as plt
import math
from in2thedeep.OM.optim_infos import OptimInfos
from in2thedeep.FFN.network import NetworkTester
import numpy as np
from in2thedeep.Interface.wrapper import Wrapper
import theano.tensor as T
import theano
import cPickle


def load_dataset():
    """ load train,valid and test set (x, label)"""
    print "loading data..."
    with open('data/galaxy_medium.pkl') as f:
    #with open('data/galaxy_big.pkl') as f:
        datasets = cPickle.load(f)
    print "done"
    return datasets


def fake_data():
    datax = np.random.random(3 * 64 * 64)
    datax = np.array([datax for d in range(128)], dtype='float32')
    datay = np.zeros((128, 37), dtype="float32")
    return ((datax, datay), [], [])


def train():
    train_set, valid_set, test_set = load_dataset()
    #train_set, valid_set, test_set = fake_data()

    batch_size = 128

    layer1 = {
        #'W':  self.W.get_value().transpose(1, 0, 3, 2),
        #'b': np.zeros((self.filter_shape[1],), dtype=theano.config.floatX),
        #'activation': self.layer_infos['activation'],
        'pooling_on': True,
        'poolsize': (2, 2),
        'border_mode': 'valid',
        'filter_shape':  (24, 3, 8, 8),
        'image_shape': (batch_size, 3, 64, 64)
    }
  
    layer2 = {
        #'W':  self.W.get_value().transpose(1, 0, 3, 2),
        #'b': np.zeros((self.filter_shape[1],), dtype=theano.config.floatX),
        #'activation': self.layer_infos['activation'],
        'pooling_on': True,
        'poolsize': (2, 2),
        'border_mode': 'valid',
        'filter_shape':  (36, 24, 4, 4),
        'image_shape': (batch_size, 24, 28, 28)
    }
  
    layer3 = {
        #'W':  self.W.get_value().transpose(1, 0, 3, 2),
        #'b': np.zeros((self.filter_shape[1],), dtype=theano.config.floatX),
        #'activation': self.layer_infos['activation'],
        'pooling_on': False,
        'border_mode': 'valid',
        'filter_shape':  (36, 36, 3, 3),
        'image_shape': (batch_size, 36, 12, 12)
    }

    layer4 = {
        'n_in': 3600,
        'n_out': 200,
        'dropout_rate': 0.5,
        'dropout': True}

    layer5 = {
        'n_in': 200,
        'n_out': 100,
        'dropout_rate': 0.5,
        'dropout': True}

    layer6 = {
        'n_in': 100,
        'n_out': 37}

    optim_infos = {
        'method': 'gradient',
        'learning_rate': 0.4,
        'batch_size': batch_size,
        'n_epochs': 100
    }

    network_architect = NetworkArchitect()
    network_architect.add_conv_pool_layer(layer1)
    network_architect.add_conv_pool_layer(layer2)
    network_architect.add_conv_pool_layer(layer3)
    network_architect.add_dense_layer(layer4)
    network_architect.add_dense_layer(layer5)
    network_architect.add_dense_layer(layer6)

    net_trainer = Wrapper(network_architect, OptimInfos(optim_infos))
    #print data
    #raw_input()
    net_trainer.fit(train_set[0], train_set[1], valid_set[:-1], test_set[:-1])


def keep_on_learning():
    train_set, valid_set, test_set = load_dataset()
    #train_set, valid_set, test_set = fake_data()

    network_architect = NetworkArchitect()
    network_architect.load_network()
    batch_size = 128
    optim_infos = {
        'method': 'gradient',
        'learning_rate': 0.05,
        'batch_size': batch_size,
        'n_epochs': 10000
    }
 
    net_trainer = Wrapper(network_architect, OptimInfos(optim_infos))
    #print data
    #raw_input()
    net_trainer.fit(train_set[0], train_set[1], valid_set[:-1], test_set[:-1])


def show_weight():

    batch_size = 1
    network_architect = NetworkArchitect()
    network_architect.load_network()
    network_architect.from_autoencoder()
    network_architect.change_batch_size(batch_size)
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the data is presented as rasterized images
    net = network_architect.build_weight_viewer(x)

    print net
    network_tester = NetworkTester(net, y)
    print network_architect.dataset_shape
    shape = network_architect.dataset_shape
    size = np.prod(shape[1:])
    print size
    raw_input()
    train_set_x = theano.shared(np.identity(size, dtype="float32"), borrow=True)
    train_set_y = theano.shared(np.identity(size, dtype="float32"), borrow=True)

    predict = network_tester.predict()
    #tester.predict()
    index = T.lscalar()    # index to a [mini]batch
    eval = theano.function([index], predict,
        givens={
            network_tester.input: train_set_x[index * batch_size:(index + 1) * batch_size],
            network_tester.y_values: train_set_y[index * batch_size:(index + 1) * batch_size]},
            on_unused_input='warn')
    n_weights = int(math.sqrt(size))
    image = []
    for i in range(n_weights):
        im_temp = (eval(i).reshape((30, 30, 1)))
        for j in range(1, n_weights):
            im = eval(i * n_weights + j).reshape((30, 30, 1))
            im_temp = np.concatenate((im, im_temp), 1)
        image.append(im_temp)
    im = np.concatenate(image, 0)

    print "shape, ", im.shape
    if im.shape[2] == 1:
        im = im.reshape(im.shape[:2])
    print im.shape
    #    im = im.reshape((30, 30))
    #im = Image.fromarray(np.uint8(255.0 * im))
    plt.imshow(np.uint8(255.0 * im),cmap=plt.cm.gray)
    plt.show()

if __name__ == "__main__":
    keep_on_learning()
    #train()
    #show_weight()
