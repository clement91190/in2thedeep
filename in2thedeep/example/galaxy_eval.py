from in2thedeep.Interface.network_architect import NetworkArchitect 
from in2thedeep.OM.optim_infos import OptimInfos
from in2thedeep.Interface.wrapper import Wrapper
import cPickle


def load_dataset():
    """ load train,valid and test set (x, label)"""
    print "loading data..."
    with open('data/galaxy_small.pkl') as f:
    #with open('data/galaxy_big.pkl') as f:
        datasets = cPickle.load(f)
    print "done"
    return datasets

def test():
    train_set, valid_set, test_set = load_dataset()
    #train_set, valid_set, test_set = fake_data()

    network_architect = NetworkArchitect()
    network_architect.load_network()
    batch_size = 128
    optim_infos = {
        'method': 'gradient',
        'learning_rate': 0.1,
        'batch_size': batch_size,
        'n_epochs': 10000
    }
 
    net_trainer = Wrapper(network_architect, OptimInfos(optim_infos))
    #print data
    #raw_input()
    net_trainer.eval(train_set[0], train_set[1])


if __name__ == "__main__":
    test()
    #train()
    #
