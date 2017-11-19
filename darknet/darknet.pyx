# cython: infer_types=True
from __future__ import print_function
from libc.stdlib cimport calloc, free
from cymem.cymem cimport Pool
import numpy
from pathlib import Path


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

    def __init__(self, bytes loc):
        if not Path(loc).exists():
            raise IOError("Metadata file not found: %s" % loc)
        self.c = get_metadata(<char*>loc)

    def __dealloc__(self):
        free_ptrs(<void**>self.c.names, self.c.classes)


cdef class Network:
    cdef network* c

    def __init__(self):
        self.c = NULL

    def __dealloc__(self):
        if self.c != NULL:
            free_network(self.c)

    @classmethod
    def load(cls, bytes cfg, bytes weights, int clear):
        if not Path(cfg).exists():
            raise IOError("Config file not found: %s" % cfg)
        if not Path(weights).exists():
            raise IOError("Weights file not found: %s" % cfg)
        cdef Network self = Network.__new__(cls)
        self.c = load_network(<char*>cfg, <char*>weights, clear)
        return self

    def detect(self, bytes loc, bytes meta_loc,
            float thresh=.5, float hier_thresh=.5, float nms=.45):
        print(loc)
        cdef Metadata meta = Metadata(meta_loc)
        cdef Image im = Image.load_color(loc, 0, 0)
        cdef box* boxes = make_boxes(self.c)
        cdef float** probs = make_probs(self.c)
        num   = num_boxes(self.c)
        network_detect(self.c, im.c, thresh, hier_thresh, nms, boxes, probs)
        res = []
        for j in range(num):
            for i in range(meta.c.classes):
                if probs[j][i] > 0:
                    res.append((meta.c.names[i], probs[j][i],
                               (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        free_ptrs(<void**>probs, num)
        free(boxes)
        return res
