import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from in2thedeep.FFN.layers.mlp_layer import HiddenLayer
from in2thedeep.FFN.network import NetworkTester, AutoEncoderBuilder 
from in2thedeep.FFN.layer import LayerBuilder


class Gradient():
    def __init__(self, datasets, network_tester, learning_rate=0.1, batch_size=20):
        self.learning_rate = learning_rate
        self.network_tester = network_tester
        self.batch_size = batch_size
        self.train_set_x, self.train_set_y = datasets
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.train_set_x.shape[0] / self.batch_size
        print self.n_train_batches

        self.rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))

        self.cost, self.updates = self.network_tester.get_cost_updates(self.learning_rate)
        self.predict = self.network_tester.predict()

        self.index = T.lscalar()    # index to a [mini]batch
        self.train_net = theano.function(
            [self.index], self.cost, updates=self.updates,
            givens={
                self.network_tester.input: self.train_set_x[self.index * self.batch_size:(self.index + 1) * self.batch_size],
                self.network_tester.y_values: self.train_set_y[self.index * self.batch_size:(self.index + 1) * self.batch_size]})
        self.eval = theano.function([self.index], self.cost,
            givens={
                self.network_tester.input: self.train_set_x[self.index * self.batch_size:(self.index + 1) * self.batch_size],
                self.network_tester.y_values: self.train_set_y[self.index * self.batch_size:(self.index + 1) * self.batch_size]})
           # allocate symbolic variables for the data

    def learning_step(self, epoch):
        print "step"
        c = []
        #for batch_index in xrange(self.n_train_batches):
        for batch_index in xrange(1):
            #print self.eval(batch_index)
            c.append(self.train_net(batch_index))
        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    def loop(self, n_epoch):
        for i in xrange(n_epoch):
            self.learning_step(i)


def main():
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the data is presented as rasterized images
    
    dataset_x = [[int(i > 5) for i in range(10)] for i in range(60)]
    dataset_x = np.array(dataset_x, dtype="float32")
    dataset_y = np.copy(dataset_x)
    dataset_x = T.shared(dataset_x) 
    dataset_x = T.shared(dataset_y) 
    datasets = (dataset_x, dataset_x)
    n_in = 10
    n_out = 2

    print  "building Net"
    layer_builder = [LayerBuilder(HiddenLayer, (n_in, n_out, None), (None, None))]
    network = AutoEncoderBuilder(layer_builder).get_network(x)
    tester = NetworkTester(network, y)
    print "...done"
    print network

    algo = Gradient(datasets, tester, 3.0)
    algo.loop(100)


if __name__ == "__main__":
    main()
