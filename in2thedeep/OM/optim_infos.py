from in2thedeep.OM.gradient import Gradient


class OptimInfos():
    """ class with all the parameters to use a learning algorithm """
    def __init__(self, infos=None):
        self.infos = {
            'method': 'gradient',
            'learning_rate': 0.1,
            'batch_size': 20,
            'n_epochs': 100,
            'take': False,
            'cost_method': 'cross'
        }
        if infos is not None:
            for key, value in infos.iteritems():
                self.infos[key] = value

    def def_method(self, datasets, network_tester):
        if self.infos['method'] == 'gradient':
            print "method : gradient"
            self.method = Gradient(datasets, network_tester, self.infos['learning_rate'], self.infos['batch_size'])
        else:
            raise NotImplementedError("unknown method")

    def run_method(self, datasets, network_tester):
        print "..building trainer"
        self.def_method(datasets, network_tester)
        print "...done"
        try:
            n_epochs = self.infos['n_epochs']
        except:
            raise NotImplementedError("need to provide a number of epochs")
        self.method.loop(n_epochs)
