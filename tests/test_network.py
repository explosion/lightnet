from __future__ import unicode_literals
from lightnet import Network

def test_init():
    nn = Network()

def test_load():
    net = Network.load(b"tiny-yolo")

def test_detect():
    net = Network.load(b"tiny-yolo")
    r = net.detect(b"example-images/dog.jpg")
    print(r)
 
