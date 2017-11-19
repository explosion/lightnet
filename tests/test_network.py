from __future__ import unicode_literals
from lightnet import Network

def test_init():
    nn = Network()

def test_load():
    net = Network.load(b"_src/cfg/tiny-yolo.cfg", b"tiny-yolo.weights", 0)

def test_detect():
    net = Network.load(b"_src/cfg/tiny-yolo.cfg", b"tiny-yolo.weights", 0)
    r = net.detect(b"_src/data/dog.jpg", b"_src/cfg/coco.data")
    print(r)
 
