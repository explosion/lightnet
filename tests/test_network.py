from __future__ import unicode_literals
from lightnet import Network, Image

def test_init():
    nn = Network()

def test_load():
    net = Network.load("tiny-yolo")

def test_detect():
    net = Network.load("tiny-yolo")
    image = Image.load_color("example-images/dog.jpg")
    r = net(image)
    print(r)
 
