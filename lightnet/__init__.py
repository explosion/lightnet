from .lightnet import Network, Image, BoxLabels
from .about import __version__

def load(name, path=None):
    return Network.load(name, path=path)
