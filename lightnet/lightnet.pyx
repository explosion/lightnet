# cython: infer_types=True
from __future__ import print_function
from libc.stdlib cimport calloc, free, rand
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
    void fill_truth_detection(char *path, int num_boxes, float *truth,
                              int classes, int flip,
                              float dx, float dy, float sx, float sy)


cdef data load_data_detection(int n, char **paths,
        int m, int w, int h, int boxes, int classes,
        float jitter, float hue, float saturation, float exposure) nogil:
    cdef data d
    d.shallow = 0

    d.X.rows = n
    d.X.vals = <float**>calloc(d.X.rows, sizeof(float*))
    d.X.cols = h*w*3

    d.y = make_matrix(n, 5*boxes)
    cdef float dw, dh, nw, nh, dx, dy
    for i in range(n):
        orig = load_image_color(paths[i], 0, 0)
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
        d.X.vals[i] = sized.data

        fill_truth_detection(paths[i], boxes, d.y.vals[i], classes,
                             flip, -dx/w, -dy/h, nw/w, nh/h)
        free_image(orig)
    return d


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


cdef class DetectionData:
    cdef data c

    def __init__(self, Network net):
        # I *think* the data from the load_args struct here gets taken over
        # by the data returned by load_data_detection. That's why we don't
        # have to free it.
        cdef load_args a = get_base_args(net.c)
        self.c = load_data_detection(a.n, a.paths,
            a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue,
            a.saturation, a.exposure)

    def __dealloc__(self):
        free_data(self.c)

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

    #def update(self, images, labels):
    #    load_args args = {0};
    #    args.w = net->w;
    #    args.h = net->h;
    #    args.paths = paths;
    #    args.n = imgs;
    #    args.m = plist->size;
    #    args.classes = classes;
    #    args.jitter = jitter;
    #    args.num_boxes = side;
    #    args.d = &buffer;
    #    args.type = REGION_DATA;

    #    args.angle = net->angle;
    #    args.exposure = net->exposure;
    #    args.saturation = net->saturation;
    #    args.hue = net->hue;
    #    
    #    args.d = &data
    #    args.type = REGION_DATA;
    #    args.angle = net->angle;
    #    args.exposure = net->exposure;
    #    args.saturation = net->saturation;
    #    args.hue = net->hue;

    #    pthread_t load_thread = load_data_in_thread(args);
    #    data = load_data_detection(len(images), paths, m, w, boxes, classes, jitter, hue, saturation, exposure)
    #    loss = train_network(self.c, train)
        
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
