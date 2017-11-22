from .lightnet import Network, Image, BoxLabels

def load(name, path=None):
    return Network.load(name, path=path)
