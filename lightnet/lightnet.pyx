# cython: infer_types=True
# cython: cdivision=True
from __future__ import print_function
from libc.stdlib cimport calloc, free, rand
from libc.string cimport memcpy
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

    def __init__(self, float[:, :, ::1] data): 
        self.c = make_image(data.shape[0], data.shape[1], data.shape[2])
        memcpy(self.c.data, &data[0,0,0], data.size * sizeof(float))

    @classmethod
    def random(cls, int w, int h, int c):
        cdef Image self = Image.__new__(cls)
        self.c = make_random_image(w, h, c)
        return self

    @classmethod
    def blank(cls, int w, int h, int c):
        cdef Image self = Image.__new__(cls)
        self.c = make_image(w, h, c)
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


cdef class Boxes:
    cdef box* c
    cdef int n

    def __init__(self, int n):
        self.c = <box*>calloc(n, sizeof(box))


cdef class BoxLabels:
    cdef box_label* c
    cdef int n

    def __init__(self, int[::1] ids, float[:, ::1] data):
        assert ids.shape[0] == data.shape[0]
        assert data.shape[1] == 4
        self.c = <box_label*>calloc(ids.shape[0], sizeof(box_label))
        self.n = ids.shape[0]
        for i in range(ids.shape[0]):
            self.c[i].id = ids[i]
        for i in range(data.shape[0]):
            self.c[i].x = data[i, 0]
            self.c[i].y = data[i, 1]
            self.c[i].h = data[i, 2]
            self.c[i].w = data[i, 3]
            self.c[i].left = self.c[i].x - self.c[i].w/2
            self.c[i].right = self.c[i].x + self.c[i].w/2
            self.c[i].top = self.c[i].y - self.c[i].h/2
            self.c[i].bottom = self.c[i].y + self.c[i].h/2

    @classmethod
    def load(cls, path):
        cdef bytes loc = unicode(Path(path).resolve()).encode('utf8')
        cdef BoxLabels self = BoxLabels.__new__(cls)
        self.c = read_boxes(loc, &self.n)
        return self

    def __dealloc__(self):
        if self.c != NULL:
            free(self.c)
        self.c = NULL

    @property
    def x(self):
        return [self.c[i].x for i in range(self.n)]

    @property
    def y(self):
        return [self.c[i].y for i in range(self.n)]

    @property
    def h(self):
        return [self.c[i].h for i in range(self.n)]

    @property
    def w(self):
        return [self.c[i].w for i in range(self.n)]

    @property
    def left(self):
        return [self.c[i].left for i in range(self.n)]

    @property
    def right(self):
        return [self.c[i].right for i in range(self.n)]

    @property
    def top(self):
        return [self.c[i].top for i in range(self.n)]

    @property
    def bottom(self):
        return [self.c[i].bottom for i in range(self.n)]


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

        cdef Image py_image
        cdef BoxLabels py_boxes
        cdef float dw, dh, nw, nh, dx, dy
        cdef float* truth
        for i, (py_image, py_boxes) in enumerate(zip(images, labels)):
            orig = py_image.c
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
            self.c.X.vals[i] = sized.data
            
            #fill_truth_detection(random_paths[i], boxes, d.y.vals[i], classes,
            #   flip -dx/w, -dy/h, nw/w, nh/h);
            # void fill_truth_detection(char *path, int num_boxes, float *truth, int classes,
            #   int flip, float dx, float dy, float sx, float sy)
            boxes = py_boxes.c
            count = py_boxes.n
            randomize_boxes(boxes, count)
            correct_boxes(boxes, count, -dx/w, -dy/h, nw/w, nh/h, flip)
            count = min(count, num_boxes)
            truth = self.c.y.vals[i]
            for j in range(count):
                box = boxes[j]
                if (box.w < .001 or box.h < .001):
                    continue
                truth[j*5+0] = box.x
                truth[j*5+1] = box.y
                truth[j*5+2] = box.w
                truth[j*5+3] = box.h
                truth[j*5+4] = box.id
 
    #def __dealloc__(self):
    #    free_data(self.c)

    @property
    def Xs(self):
        output = []
        for i in range(self.c.X.rows):
            vals = [self.c.X.vals[i][j] for j in range(self.c.X.cols)]
            output.append(vals)
        return output
    
    @property
    def ys(self):
        output = []
        for i in range(self.c.y.rows):
            vals = [self.c.y.vals[i][j] for j in range(self.c.y.cols)]
            output.append(vals)
        return output

    @property
    def X_shape(self):
        return (self.c.X.rows, self.c.X.cols)
    
    @property
    def y_shape(self):
        return (self.c.y.rows, self.c.y.cols)


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

    #def __dealloc__(self):
    #    free_ptrs(<void**>self.c.names, self.c.classes)
    #    shutil.rmtree(self.backup_dir)


cdef class Network:
    cdef network* c
    cdef Metadata meta

    def __init__(self):
        self.c = NULL

    #def __dealloc__(self):
    #    if self.c != NULL:
    #        free_network(self.c)

    @property
    def num_boxes(self):
        return num_boxes(self.c)

    @property
    def width(self):
        return network_width(self.c)
    
    @property
    def height(self):
        return network_height(self.c)

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
        return self._detect(image, thresh, hier_thresh, nms)

    def update(self, images, labels):
        cdef DetectionData data
        cdef int max_boxes = self.c.layers[self.c.n-1].max_boxes
        data = DetectionData(images, labels,
                self.width, self.height, max_boxes)
        cdef float loss = 0.
        assert data.c.X.rows % self.c.batch == 0
        cdef int batch = self.c.batch
        cdef int n = data.c.X.rows / batch

        loss = train_network(self.c, data.c)
        return loss

    def _detect(self, Image image,
            float thresh=.5, float hier_thresh=.5, float nms=.45):
        num = num_boxes(self.c)
        cdef Boxes boxes = Boxes(num)
        cdef float** probs = make_probs(self.c)
        network_detect(self.c, image.c, thresh, hier_thresh, nms, boxes.c, probs)
        res = []
        cdef int j, i
        for j in range(num):
            for i in range(self.meta.c.classes):
                if probs[j][i] > 0:
                    res.append((i, self.meta.c.names[i], probs[j][i],
                               (boxes.c[j].x, boxes.c[j].y,
                                boxes.c[j].w, boxes.c[j].h)))
        res = sorted(res, key=lambda x: -x[2])
        free_ptrs(<void**>probs, num)
        return res
