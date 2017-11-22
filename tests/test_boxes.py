import pytest
import numpy
from lightnet.lightnet import BoxLabels

@pytest.fixture
def ids():
    return numpy.asarray([0, 2], dtype='i')

@pytest.fixture
def xywh():
    return numpy.asarray(
        [[2., 2., 2., 2.],
        [1., 1., 1., 1.]], dtype='f')


def test_BoxLabels_init(ids, xywh):
    labels = BoxLabels(ids, xywh)
    assert labels.x == list(xywh[:, 0])
    assert labels.y == list(xywh[:, 1])
    assert labels.h == list(xywh[:, 2])
    assert labels.w == list(xywh[:, 3])
    print(labels.left)
    print(labels.right)
    print(labels.top)
    print(labels.bottom)
