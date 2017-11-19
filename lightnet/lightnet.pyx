# cython: infer_types=True
from __future__ import print_function
from libc.stdlib cimport calloc, free, rand
cimport numpy as np
import shutil
import tempfile
import numpy
from pathlib import Path


try:
    unicode
except NameError:
    unicode = str

cdef extern from "_darknet/utils.h" nogil:
    float rand_uniform(float min, float max)

cdef extern from "_darknet/image.h" nogil:
    void place_image(image im, int w, int h, int dx, int dy, image canvas)

cdef extern from "_darknet/data.h" nogil:
    void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
    void randomize_boxes(box_label *b, int n)


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
    def load_color(cls, path, int w=0, int h=0):
        path = Path(path)
        if not path.exists():
            raise IOError("Color image not found: %s" % path)
        cdef Image self = Image.__new__(cls)
        cdef bytes loc = unicode(path.resolve()).encode('utf8')
        self.c = load_image_color(<char*>loc, w, h)
        return self

    def __dealloc__(self):
        free_image(self.c)


cdef class BoxLabels:
    cdef box_label* c
    cdef int n

    def __init__(self, path=None):
        cdef bytes loc = unicode(Path(path).resolve()).encode('utf8')
        self.c = read_boxes(loc, &self.n)

    def __dealloc__(self):
        free(self.c)


cdef class DetectionData:
    cdef data c

    def __init__(self, images, labels, int w, int h, int num_boxes,
            float jitter=0.2, float hue=0.1, float saturation=1.5, float exposure=1.5):
        self.c.shallow = 0
        cdef int n = len(images)
        self.c.X.rows = n
        self.c.X.vals = <float**>calloc(self.c.X.rows, sizeof(float*))
        self.c.X.cols = h*w*3
        self.c.y = make_matrix(n, 5*num_boxes)

        cdef Image image
        cdef BoxLabels boxes
        for i, (image, boxes) in enumerate(zip(images, labels)):
            self._fill_Xy_vals(&self.c.X.vals[i], self.c.y.vals[i],
                image.c, boxes.c, num_boxes,
                w, h, jitter, hue, saturation, exposure)
 
    def __dealloc__(self):
        free_data(self.c)

    cdef float* _fill_Xy_vals(self, float** X, float* truth,
            image orig, box_label* boxes, int num_boxes, int w, int h,
            float jitter, float hue, float saturation, float exposure) nogil:
        cdef float dw, dh, nw, nh, dx, dy
        sized = make_image(w, h, orig.c)
        fill_image(sized, .5)

        dw = jitter * orig.w
        dh = jitter * orig.h

        new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh))
        scale = rand_uniform(.25, 2)

        if new_ar < 1:
            nh = scale * h
            nw = nh * new_ar
        else:
            nw = scale * w
            nh = nw / new_ar

        dx = rand_uniform(0, w - nw)
        dy = rand_uniform(0, h - nh)

        place_image(orig, <int>nw, <int>nh, <int>dx, <int>dy, sized)

        random_distort_image(sized, hue, saturation, exposure)

        flip = rand() % 2
        if flip:
            flip_image(sized)
        X[0] = sized.data
        randomize_boxes(boxes, num_boxes)
        correct_boxes(boxes, num_boxes, -dx/w, -dy/h, nw/w, nh/h, flip)
        for i in range(num_boxes):
            box = boxes[i]
            if (box.w < .001 or box.h < .001):
                continue
            truth[i*5+0] = box.x
            truth[i*5+1] = box.y
            truth[i*5+2] = box.w
            truth[i*5+3] = box.h
            truth[i*5+4] = box.id


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

    @property
    def num_boxes(self):
        return num_boxes(self.c)

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

    def __call__(self, Image image, 
            float thresh=.5, float hier_thresh=.5, float nms=.45):
        return self.detect(image, thresh, hier_thresh, nms)

    def detect(self, Image image,
            float thresh=.5, float hier_thresh=.5, float nms=.45):
        cdef box* boxes = make_boxes(self.c)
        cdef float** probs = make_probs(self.c)
        num = num_boxes(self.c)
        network_detect(self.c, image.c, thresh, hier_thresh, nms, boxes, probs)
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
