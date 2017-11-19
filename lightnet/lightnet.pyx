# cython: infer_types=True
from __future__ import print_function
from libc.stdlib cimport calloc, free
import shutil
import tempfile
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
    def load(cls, path, int w, int h, int c):
        path = Path(path)
        if not path.exists():
            raise IOError("Image not found: %s" % path)
        cdef Image self = Image.__new__(cls)
        cdef bytes loc = unicode(path.resolve()).encode('utf8')
        self.c = load_image(<char*>loc, w, h, c)
        return self
    
    @classmethod
    def load_color(cls, path, int w, int h):
        path = Path(path)
        if not path.exists():
            raise IOError("Color image not found: %s" % path)
        cdef Image self = Image.__new__(cls)
        cdef bytes loc = unicode(path.resolve()).encode('utf8')
        self.c = load_image_color(<char*>loc, w, h)
        return self

    def __dealloc__(self):
        free_image(self.c)


cdef class Metadata:
    cdef metadata c
    cdef public object backup_dir

    def __init__(self, template_path):
        template_path = Path(template_path)
        if not template_path.exists():
            raise IOError("Metadata template not found: %s" % template_path)
        with template_path.open('r', encoding='utf8') as file_:
            text = file_.read()
        data_dir = Path(__file__).parent / 'data'
        self.backup_dir = tempfile.mkdtemp()
        text = text.replace('$DATA', str(data_dir.resolve()))
        text = text.replace('$HERE', str(data_dir.resolve()))
        text = text.replace('$BACKUP', self.backup_dir)
        out_loc = Path(str(template_path).replace('.template', '.data'))
        with out_loc.open('w', encoding='utf8') as file_:
            file_.write(text)
        cdef bytes loc = unicode(out_loc.resolve()).encode('utf8')
        self.c = get_metadata(<char*>loc)

    def __dealloc__(self):
        free_ptrs(<void**>self.c.names, self.c.classes)
        shutil.rmtree(self.backup_dir)


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
    def load(cls, name, *, path=None, int clear=0):
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
        self.load_meta(path / 'coco.template')
        return self

    def __call__(self, loc, 
            float thresh=.5, float hier_thresh=.5, float nms=.45):
        return self.detect(loc, thresh, hier_thresh, nms)

    def detect(self, loc,
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
