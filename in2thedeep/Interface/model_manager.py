import pickle
from in2thedeep.FFN.network import NetworkTester
#TODO make this useful see how to mix with NetworkTester


class ModelManager:
    def __init__(self, path, network):
        self.path = path

    def load(self):
        with open(self.path, 'r') as fich:
            self.network_infos = pickle.load(fich)

    def save(self):
        with open(self.path, 'w') as fich:
            pickle.dump(self.path, fich)


