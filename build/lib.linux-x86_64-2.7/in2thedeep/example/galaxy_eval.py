from in2thedeep.Interface.network_architect import NetworkArchitect 
from in2thedeep.OM.optim_infos import OptimInfos
from in2thedeep.Interface.wrapper import Wrapper
import cPickle


def load_dataset():
    """ load train,valid and test set (x, label)"""
    print "loading data..."
    #with open('data/galaxy_small.pkl') as f:
    with open('data/galaxy_big.pkl') as f:
        datasets = cPickle.load(f)
    print "done"
    return datasets


def test():
    train_set, valid_set, test_set = load_dataset()
    #train_set, valid_set, test_set = fake_data()


def transform(set):
    network_architect = NetworkArchitect()
    network_architect.load_network("model_save.tkl")

    network_architect.list_of_layers_infos.pop()
    network_architect.list_of_layers_infos.pop()
    network_architect.list_of_layers_infos.pop()
    #net = FFNetwork(input, network_architect.list_of_layers_infos[:-3], network_architect.dataset_shape, network_architect.input_type)

    batch_size = 1000
    optim_infos = {
        'method': 'gradient',
        'learning_rate': 0.1,
        'batch_size': batch_size,
        'n_epochs': 10000
    }

    net_trainer = Wrapper(network_architect, OptimInfos(optim_infos))
    #print data
    #raw_input()
    return net_trainer.transform(set)


def save_train(path="conv_train.pkl"):

    train_set, valid_set, test_set = load_dataset()
    train_set = list(train_set)
    valid_set = list(valid_set)
    test_set = list(test_set)
    train_set[0] = transform(train_set[0])
    valid_set[0] = transform(valid_set[0])
    test_set[0] = transform(test_set[0])
    
    with open(path, 'w') as fich:
        cPickle.dump((train_set, valid_set, test_set), fich)


if __name__ == "__main__":
    #test()
    save_train()
    #train()
    #
