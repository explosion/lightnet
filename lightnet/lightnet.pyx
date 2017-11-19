# cython: infer_types=True
from __future__ import print_function
from libc.stdlib cimport calloc, free
import numpy
from pathlib import Path

try:
    unicode
except NameError:
    unicode = str


cdef class Image:
    cdef image c

    def __init__(self, int w, int h, int c): 
        self.c = make_image(w, h, c)

    @classmethod
    def random(cls, int w, int h, int c):
        cdef Image self = Image.__new__(cls)
        self.c = make_random_image(w, h, c)
        return self

    @classmethod
    def load(cls, bytes loc, int w, int h, int c):
        if not Path(loc).exists():
            raise IOError("Image not found: %s" % loc)
        cdef Image self = Image.__new__(cls)
        self.c = load_image(<char*>loc, w, h, c)
        return self
    
    @classmethod
    def load_color(cls, bytes loc, int w, int h):
        if not Path(loc).exists():
            raise IOError("Color image not found: %s" % loc)
        cdef Image self = Image.__new__(cls)
        self.c = load_image_color(<char*>loc, w, h)
        return self

    def __dealloc__(self):
        free_image(self.c)


cdef class Metadata:
    cdef metadata c

    def __init__(self, path):
        if not Path(path).exists():
            raise IOError("Metadata file not found: %s" % path)
        cdef bytes loc = unicode(path.resolve()).encode('utf8')
        self.c = get_metadata(<char*>loc)

    def __dealloc__(self):
        free_ptrs(<void**>self.c.names, self.c.classes)


cdef class Network:
    cdef network* c
    cdef Metadata meta

    def __init__(self):
        self.c = NULL

    def __dealloc__(self):
        if self.c != NULL:
            free_network(self.c)

    def load_meta(self, path):
        self.meta = Metadata(path)

    @classmethod
    def load(cls, bytes name, *, path=None, int clear=0):
        if path is None:
            path = Path(__file__).parent / 'data'
        path = Path(path)
        if not path.exists():
            raise IOError("Data path not found: %s" % path)
        cfg_path = path / '{name}.cfg'.format(name=name)
        weights_path = path / '{name}.weights'.format(name=name)
        if not cfg_path.exists():
            raise IOError("Config file not found: %s" % cfg_path)
        if not weights_path.exists():
            raise IOError("Weights file not found: %s" % weights_path)
        cdef Network self = Network.__new__(cls)
        cdef bytes cfg = unicode(cfg_path.resolve()).encode('utf8')
        cdef bytes weights = unicode(weights_path.resolve()).encode('utf8')
        self.c = load_network(<char*>cfg, <char*>weights, clear)
        # TODO: Fix this hard-coding...
        self.load_meta(path / 'coco.data')
        return self

    def detect(self, bytes loc,
            float thresh=.5, float hier_thresh=.5, float nms=.45):
        cdef Image im = Image.load_color(loc, 0, 0)
        cdef box* boxes = make_boxes(self.c)
        cdef float** probs = make_probs(self.c)
        num = num_boxes(self.c)
        network_detect(self.c, im.c, thresh, hier_thresh, nms, boxes, probs)
        res = []
        cdef int j, i
        for j in range(num):
            for i in range(self.meta.c.classes):
                if probs[j][i] > 0:
                    res.append((self.meta.c.names[i], probs[j][i],
                               (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        free_ptrs(<void**>probs, num)
        free(boxes)
        return res
