from __future__ import unicode_literals
from lightnet import Network, Image, BoxLabels
from lightnet.lightnet import DetectionData
import numpy
import pytest

@pytest.fixture
def ids_xywh():
    return (numpy.asarray([16], dtype='i'),
            numpy.asarray([[0.606688, 0.341381, 0.544156, 0.510000]], dtype='f'))

@pytest.fixture
def image():
    return Image.load_color("_src/coco/images/val2014/COCO_val2014_000000000042.jpg")

@pytest.fixture
def box_labels(ids_xywh):
    ids, xywh = ids_xywh
    return BoxLabels(ids, xywh)
 
def test_init():
    nn = Network()

def test_load():
    net = Network.load("tiny-yolo")

def test_detect(image):
    net = Network.load("tiny-yolo")
    #assert net(image)
 
def test_box_labels(box_labels):
    pass
 

def test_detection_data(image, box_labels):
    net = Network.load("tiny-yolo")
    data = DetectionData([image], [box_labels],
                         net.width, net.height, net.num_boxes)
    assert data.X_shape == (1, net.width * net.height * 3)
    assert data.y_shape == (1, net.num_boxes * 5)

def test_update(image, box_labels):
    net = Network.load("tiny-yolo")
    for i in range(10):
        loss = net.update([image], [box_labels])
        print(loss)
