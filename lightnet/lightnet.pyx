# cython: infer_types=True
# cython: cdivision=True
from __future__ import print_function
from libc.stdlib cimport calloc, free, rand
from libc.string cimport memcpy, memset
cimport numpy as np
import shutil
import tempfile
import numpy
from pathlib import Path
import json
import msgpack

from .util import make_temp_dir


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


cdef extern from "_darknet/stb_image.h" nogil:
    ctypedef unsigned char stbi_uc
    stbi_uc *stbi_load_from_memory(const stbi_uc* raw, int length, int *x,
                                   int *y, int *comp, int req_comp)

cdef class Image:
    cdef image c

    def __init__(self, float[:, :, ::1] data): 
        self.c = make_image(data.shape[0], data.shape[1], data.shape[2])
        memcpy(self.c.data, &data[0,0,0], data.size * sizeof(float))

    @property
    def data(self):
        cdef np.ndarray array = numpy.zeros((self.c.w, self.c.h, self.c.c),
                                             dtype='f')
        memcpy(<float*>array.data, self.c.data, self.c.h*self.c.w*self.c.c*sizeof(float))
        return array

    @property
    def width(self):
        return self.c.w

    @property
    def height(self):
        return self.c.h

    @classmethod
    def from_bytes(cls, bytes raw, int channels=3):
        cdef stbi_uc* img_data
        cdef int w, h, c
        img_data = stbi_load_from_memory(<stbi_uc*>raw, len(raw),
                        &w, &h, &c, channels)
        if channels:
            c = channels
        cdef Image self = Image.blank(w, h, c)
        cdef int k, j, i, dst_index, src_index
        for k in range(c):
            for j in range(h):
                for i in range(w):
                    dst_index = i + w*j + w*h*k
                    src_index = k + c*i + c*w*j
                    self.c.data[dst_index] = <float>img_data[src_index] / 255.
        return self
 
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


def get_relative_box(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def get_absolute_box(size, box):
    raise NotImplementedError


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

    def __init__(self, images, labels, int w, int h, int max_boxes, int classes,
            float jitter=0.2, float hue=0.1, float saturation=1.5, float exposure=1.5):
        self.c.shallow = 0
        cdef int n = len(images)
        self.c.X.rows = n
        self.c.X.vals = <float**>calloc(sizeof(float*), self.c.X.rows)
        self.c.X.cols = h*w*3
        self.c.y = make_matrix(n, 5 * max_boxes)

        cdef Image py_image
        cdef BoxLabels py_boxes
        cdef float dw, dh, nw, nh, dx, dy
        cdef float* truth
        cdef int index = 0
        cdef float pleft, pright, pwidth, pheight
        for i, (py_image, py_boxes) in enumerate(zip(images, labels)):
            self.c.X.vals[i] = _load_data_detection(self.c.y.vals[i],
                                    py_image.c, py_boxes.c, py_boxes.n, max_boxes,
                                    classes, jitter, hue, saturation, exposure)
    
    def __dealloc__(self):
        free_data(self.c)

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


cdef float* _load_data_detection(float* truths, image orig, box_label* boxes,
                                 int count, int max_boxes, int num_classes,
                                 float jitter, float hue, float saturation,
                                 float exposure) nogil:
    cdef image sized = make_image(orig.w, orig.h, orig.c)
    fill_image(sized, .5);

    cdef float dw = jitter * orig.w
    cdef float dh = jitter * orig.h

    cdef float new_ar = (orig.w + rand_uniform(-dw, dw)) / (orig.h + rand_uniform(-dh, dh));
    cdef float scale = rand_uniform(.25, 2)

    cdef float nw, nh

    if new_ar < 1:
        nh = scale * orig.h
        nw = nh * new_ar
    else:
        nw = scale * orig.w
        nh = nw / new_ar

    cdef float dx = rand_uniform(0, orig.w - nw)
    cdef float dy = rand_uniform(0, orig.h - nh)

    place_image(orig, <int>nw, <int>nh, <int>dx, <int>dy, sized)

    random_distort_image(sized, hue, saturation, exposure)

    cdef int flip = rand()%2
    if flip:
        flip_image(sized)

    _fill_truth_detection(boxes, count, max_boxes, truths, num_classes, flip,
        -dx/orig.w, -dy/orig.h, nw/orig.w, nh/orig.h)
    return sized.data

cdef void _fill_truth_detection(box_label* boxes, int count, int num_boxes,
                                float *truth, int classes, int flip,
                                float dx, float dy, float sx, float sy) nogil:
    randomize_boxes(boxes, count)
    correct_boxes(boxes, count, dx, dy, sx, sy, flip)
    if count > num_boxes:
        count = num_boxes
    cdef float x, y, w, h
    cdef int id, i

    for i in range(count):
        x =  boxes[i].x
        y =  boxes[i].y
        w =  boxes[i].w
        h =  boxes[i].h
        id = boxes[i].id

        if ((w < .001 or h < .001)):
            continue

        truth[i*5+0] = x
        truth[i*5+1] = y
        truth[i*5+2] = w
        truth[i*5+3] = h
        truth[i*5+4] = id


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
        if self.c.names != NULL:
            free_ptrs(<void**>self.c.names, self.c.classes)
            self.c.names = NULL
            shutil.rmtree(self.backup_dir)


cdef class Network:
    cdef network* c
    cdef public object cfg
    cdef readonly object names

    def __init__(self):
        self.c = NULL

    def __dealloc__(self):
        if self.c != NULL:
            free_network(self.c)
            self.c = NULL

    @property
    def num_classes(self):
        return self.c.layers[self.c.n-1].classes
    
    @property
    def num_boxes(self):
        return num_boxes(self.c)
    
    @property
    def max_boxes(self):
        return self.c.layers[self.c.n - 1].max_boxes

    @property
    def side(self):
        return self.c.layers[self.c.n - 1].side

    @property
    def width(self):
        return network_width(self.c)
    
    @property
    def height(self):
        return network_height(self.c)

    @classmethod
    def load(cls, name, *, path=None, names=None, int clear=0):
        if path is None:
            path = Path(__file__).parent / 'data'
        path = Path(path)
        if not path.exists():
            raise IOError("Data path not found: %s" % path)
        cfg_path = path / '{name}.cfg'.format(name=name)
        if name == 'yolo.2.0':
            weights_path = path / 'yolo.weights'
        else:
            weights_path = path / '{name}.weights'.format(name=name)
        if not cfg_path.exists():
            raise IOError("Config file not found: %s" % cfg_path)
        if not weights_path.exists():
            raise IOError("Weights file not found: %s" % weights_path)
        cdef Network self = Network.__new__(cls)
        cdef bytes cfg = unicode(cfg_path.resolve()).encode('utf8')
        cdef bytes weights = unicode(weights_path.resolve()).encode('utf8')
        self.cfg = cfg
        self.c = load_network(<char*>cfg, <char*>weights, clear)
        # TODO: Fix this hard-coding...
        with (path / 'coco.names').open('r', encoding='utf8') as file_:
            self.names = file_.read().split()
        return self

    def __call__(self, Image image, 
            float thresh=.5, float hier_thresh=.5, float nms=.45):
        return self._detect(image, thresh, hier_thresh, nms)

    def update(self, images, labels):
        assert len(images) != 0
        assert len(labels) == len(images)
        cdef DetectionData data
        data = DetectionData(images, labels,
                self.width, self.height, self.max_boxes, self.num_classes)
        cdef float loss = 0.
        cdef int prev_batch_size = self.c.batch
        set_batch_network(self.c, data.c.X.rows)
        resize_network(self.c, self.c.w, self.c.h)
        self.c.train = 1
        get_next_batch(data.c, self.c.batch, 0, self.c.input, self.c.truth)
        forward_network(self.c)
        backward_network(self.c)
        loss += self.c.cost[0]
        update_network(self.c)
        self.c.train = 0
        set_batch_network(self.c, prev_batch_size)
        resize_network(self.c, self.c.w, self.c.h)
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
            for i in range(len(self.names)):
                if probs[j][i] > 0:
                    res.append((i, self.names[i], probs[j][i],
                               (boxes.c[j].x, boxes.c[j].y,
                                boxes.c[j].w, boxes.c[j].h)))
        res = sorted(res, key=lambda x: -x[2])
        free_ptrs(<void**>probs, num)
        return res
    
    def to_bytes(self):
        with make_temp_dir() as temp_dir:
            self.to_disk(temp_dir)
            temp_dir = Path(temp_dir)
            msg = {}
            with (temp_dir / 'weights').open('rb') as file_:
                msg[b'weights'] = file_.read()
            with (temp_dir / 'cfg').open('rb') as file_:
                msg[b'cfg'] = file_.read()
            with (temp_dir / 'names').open('rb') as file_:
                msg[b'names'] = file_.read()
        return msgpack.dumps(msg)

    def from_bytes(self, b):
        msg = msgpack.loads(b)
        with make_temp_dir() as temp_dir:
            temp_dir = Path(temp_dir)
            with (temp_dir / 'weights').open('wb') as file_:
                file_.write(msg[b'weights'])
            with (temp_dir / 'cfg').open('wb') as file_:
                file_.write(msg[b'cfg'])
            with (temp_dir / 'names').open('wb') as file_:
                file_.write(msg[b'names'])
            self.from_disk(temp_dir)
        return self

    def to_disk(self, path):
        path = Path(path)
        cdef bytes weights_loc = unicode(path / 'weights').encode('utf8')
        save_weights(self.c, <char*>weights_loc)
        with (path / 'cfg').open('wb') as file_:
            file_.write(self.cfg)
        with (path / 'names').open('w', encoding='utf8') as file_:
            file_.write('\n'.join(self.names))

    def from_disk(self, path):
        path = Path(path)
        if not path.exists():
            raise IOError("Model path not found: %s" % path)
        if not path.is_dir():
            raise IOError("Model path not directory: %s" % path)
        if not (path / 'weights').exists():
            raise IOError("Weights path not found: %s" % (path/'weights'))
        if not (path / 'cfg').exists():
            raise IOError("Config path not found: %s" % (path/'cfg'))
        cdef bytes weights_loc = path2bytes(path / 'weights')
        cdef bytes cfg_loc = path2bytes(path / 'cfg')
        self.c = load_network(cfg_loc, weights_loc, 0)
        self.cfg = (path / 'cfg').open('rb').read()
        with (path / 'names').open('r', encoding='utf8') as file_:
            self.names = file_.read().split()
        return self


cpdef bytes path2bytes(path):
    path = Path(path)
    if not path.exists():
        raise IOError("Data path not found: %s" % path)
    return unicode(path.resolve()).encode('utf8')


def train(bytes cfgfile_, bytes weightfile_, bytes train_images_, bytes backup_directory_):
    cdef char* cfgfile = cfgfile_
    cdef char* weightfile = weightfile_
    cdef char* train_images = train_images_
    cdef char* backup_directory = backup_directory_
    cdef network* net = load_network(cfgfile, weightfile, 0)
    cdef char* base = basecfg(cfgfile)
    print("Learning Rate: %f, Momentum: %f, Decay: %f\n" % 
            (net.learning_rate, net.momentum, net.decay))
    cdef int imgs = net.batch * net.subdivisions
    cdef int i = net.seen[0]/imgs

    cdef layer l = net.layers[net.n - 1]

    cdef int side = l.side
    cdef int classes = l.classes
    cdef float jitter = l.jitter

    cdef list *plist = get_paths(train_images)
    cdef char **paths = <char**>list_to_array(plist)

    cdef load_args args
    #memset(&args, 0, sizeof(args))
    args.w = net.w
    args.h = net.h
    args.paths = paths
    args.coords = l.coords
    args.n = imgs
    args.m = plist.size
    args.classes = classes
    args.jitter = jitter
    args.num_boxes = l.max_boxes
    args.type = DETECTION_DATA
    args.angle = net.angle
    args.exposure = net.exposure
    args.saturation = net.saturation
    args.hue = net.hue

    cdef float loss
    if args.exposure == 0:
        args.exposure = 1
    if args.saturation == 0:
        args.saturation = 1
    if args.aspect == 0:
        args.aspect = 1

    cdef data train = load_data_detection(args.n, args.paths, args.m, args.w, args.h,
                        args.num_boxes, args.classes, args.jitter,
                        args.hue, args.saturation, args.exposure)

    loss = train_network(net, train)
    print(loss)
    while get_current_batch(net) < net.max_batches:
        args.d = &train
        load_data_blocking(args)
        loss = train_network(net, train)
        print(loss)
