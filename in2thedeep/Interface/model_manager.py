import pickle
from in2thedeep.FFN.network import NetworkTester
#TODO make this useful see how to mix with NetworkTester


class ModelManager:
    """ class to load and save model """
    def __init__(self, path, network):
        self.path = path

    def load(self):
        with open(self.path, 'r') as fich:
            self.network_infos = pickle.load(fich)

    def save(self):
        with open(self.path, 'w') as fich:
            pickle.dump(self.path, fich)

    def save_under(self, path):
        self.path = path
        self.save()
            
    def __str__(self):
        return self.nework_infos.__str__()

