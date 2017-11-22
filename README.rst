LightNet: Bringing pjreddie's DarkNet out of the shadows
********************************************************

LightNet provides a simple and efficient Python interface to
`DarkNet <https://github.com/pjreddie/darknet>`_, a neural  network library
written by Joseph Redmon that's well known for its state-of-the-art object
detection models, `YOLO and YOLOv2 <https://pjreddie.com/darknet/yolo/>`_.

LightNet's features include:

* **State-of-the-art object detection**: YOLOv2 offers unmatched speed/accuracy trade-offs.
* **Easy-to-use via Python**: Pass in byte strings, get back numpy arrays with bounding boxes.
* **Lightweight and self-contained**: No dependency on large frameworks like Tensorflow, PyTorch etc. The DarkNet source is provided in the package.
* **Easy to install**: Just ``pip install lightnet`` and ``python -m lightnet download yolo``.
* **Cross-platform**: Works on OSX, Linux and Windows, on Python 2.7, 3.5 and 3.6.
* **10x faster on CPU**: Uses BLAS for its matrix multiplications routines.
* **Not named DarkNet**: Avoids some potentially awkward misunderstandings.

An object detection system predicts labelled bounding boxes on an image. The
label scheme comes from the training data, so different models will have
different label sets. YOLOv2 can detect objects in images of any resolution.
Smaller images will be faster to predict, while high resolution images will
give you better object detection accuracy.

Images can be loaded by file-path, by JPEG-encoded byte-string, or by numpy
array. If passing in a numpy array, it should be of dtype float32, and shape
``(width, height, colors)``.

Installation
============

==================== ===
**Operating system** macOS / OS X, Linux, Windows (Cygwin, MinGW, Visual Studio)
**Python version**   CPython 2.7, 3.5, 3.6. Only 64 bit.
**Package managers** pip (source packages only)
==================== ===

LightNet can be installed via pip:

.. code:: bash

    pip install lightnet

Once you've downloaded LightNet, you can install a model using the
``lightnet download`` command. This will save the models in the
``lightnet/data`` directory. If you've installed LightNet system-wide, make
sure to run the command as administrator.

.. code:: bash

    python -m lightnet download tiny-yolo
    python -m lightnet download yolo

The following models are currently available via the ``download`` command:

===================== ======= ===
``yolo.weights``      258 MB  `Direct download`__
``tiny-yolo.weights`` 44.9 MB `Direct download`__
===================== ======= ===

__ https://pjreddie.com/media/files/yolo.weights
__ https://pjreddie.com/media/files/tiny-yolo.weights

Usage
=====

.. code:: python

    import lightnet

    model = lightnet.load('tiny-yolo')
    image = lightnet.Image.from_bytes(open('eagle.jpg', 'rb'))
    boxes = model(image)

``METHOD`` lightnet.load
------------------------

Load a pre-trained model. If a ``path`` is provided, it shoud be a directory
containing two files,  named ``{name}.weights`` and ``{name}.cfg``. If a
``path`` is not provided, the built-in data directory is used, which is
located within the LightNet package.

.. code:: python

    model = lightnet.load('tiny-yolo')
    model = lightnet.load(path='/path/to/yolo')

=========== =========== ===========
Argument    Type        Description
=========== =========== ===========
``name``    unicode     Name of the model located in the data directory, e.g. ``tiny-yolo``.
``path``    unicode     Optional path to a model data directory.
**RETURNS** ``Network`` The loaded model.
=========== =========== ===========

Network
=======

The neural network object. Wraps DarkNet's ``network`` struct.

``CLASSMETHOD`` Network.load
----------------------------

Load a pre-trained model. Identical to ``lightnet.load()``.

``METHOD`` Network.__call__
---------------------------

Detect bounding boxes given an ``Image`` object. The bounding boxes are
provided as a list, with each entry
``(class_id, class_name, prob, [(x, y, width, height)])``, where ```x``` and
``y``` are the pixel coordinates of the center of the centre of the box, and
``width`` and ``height`` describe its dimensions. ``class_id`` is the integer
index of the object type, class_name is a string with the object type, and
``prob`` is a float indicating the detection score. The ``thresh`` parameter
controls the prediction threshold. Objects with a detection probability above
``thresh`` are returned. We don't know what ``hier_thresh`` or ``nms`` do.

.. code:: python

    boxes = model(image, thresh=0.5, hier_thresh=0.5, nms=0.45)

=============== =========== ===========
Argument        Type        Description
=============== =========== ===========
``image``       ``Image``   The image to process.
``thresh``      float       Prediction threshold.
``hier_thresh`` float
``path``        unicode     Optional path to a model data directory.
**RETURNS**     list        The bounding boxes, as ``(class_id, class_name, prob, xywh)`` tuples.
=============== =========== ===========

``METHOD`` Network.update
-------------------------

Update the model, on a batch of examples. The images should be provided as a
list of ``Image`` objects. The ``box_labels`` should be a list of ``BoxLabel``
objects. Returns a float, indicating how much the models prediction differed
from the provided true labels.

.. code:: python

    loss = model.update([image1, image2], [box_labels1, box_labels2])

============== =========== ===========
Argument       Type        Description
============== =========== ===========
``images``     list        List of ``Image`` objects.
``box_labels`` list        List of ``BoxLabel`` objects.
**RETURNS**    float       The loss indicating how much the prediction differed from the provided labels.
============== =========== ===========

Image
=====

Data container for a single image. Wraps DarkNet's ``image`` struct.

``METHOD`` Image.__init__
-------------------------

Create an image. `data` should be a numpy array of dtype float32, and shape
(width, height, colors).

.. code:: python

    image = Image(data)

=========== =========== ===========
Argument    Type        Description
=========== =========== ===========
``data``    numpy array The image data
**RETURNS** ``Image``   The newly constructed object.
=========== =========== ===========

``CLASSMETHOD`` Image.blank
---------------------------

Create a blank image, of specified dimensions.

.. code:: python

    image = Image.blank(width, height, colors)

=========== =========== ===========
Argument    Type        Description
=========== =========== ===========
``width``   int         The image width, in pixels.
``height``  int         The image height, in pixels.
``colors``  int         The number of color channels (usually ``3``).
**RETURNS** ``Image``   The newly constructed object.
=========== =========== ===========

``CLASSMETHOD`` Image.load
--------------------------

Load an image from a path to a jpeg file, of the specified dimensions.

.. code:: python

    image = Image.load(path, width, height, colors)

=========== =========== ===========
Argument    Type        Description
=========== =========== ===========
``path``    unicode     The path to the image file.
``width``   int         The image width, in pixels.
``height``  int         The image height, in pixels.
``colors``  int         The number of color channels (usually ``3``).
**RETURNS** ``Image``   The newly constructed object.
=========== =========== ===========

``CLASSMETHOD`` Image.from_bytes
--------------------------------

Read an image from a byte-string, which should be the contents of a jpeg file.

.. code:: python

    image = Image.from_bytes(bytes_data)

============== =========== ===========
Argument       Type        Description
============== =========== ===========
``bytes_data`` bytes       The image contents.
**RETURNS**    ``Image``   The newly constructed object.
============== =========== ===========

BoxLabels
=========

Data container for labelled bounding boxes for a single image. Wraps an array
of DarkNet's ``box_label`` struct.

``METHOD`` BoxLabels.__init__
-----------------------------

Labelled box annotations for a single image, used to update the model. ``ids``
should be a 1d numpy array of dtype int32, indicating the correct class IDs of
the objects. ``boxes`` should be a 2d array of dtype float32, and shape
``(len(ids), 4)``. The 4 columns of the boxes should provide the **relative**
``x, y, width, height`` of the bounding box, where ``x`` and ``y`` are the
coordinates of the centre, relative to the image size, and ``width`` and
``height`` are the relative dimensions of the box.

.. code:: python

    box_labels = BoxLabels(ids, boxes)

============== ============= ===========
Argument       Type          Description
============== ============= ===========
``ids``        numpy array   The class IDs of the objects.
``boxes``      numpy array   The boxes providing the relative ``x, y, width, height`` of the bounding box.
**RETURNS**    ``BoxLabels`` The newly constructed object.
============== ============= ===========

``CLASSMETHOD`` BoxLabels.load
------------------------------

Load annotations for a single image from a text file. Each box should be
described on a single line, in the format ``class_id x y width height``.

.. code:: python

    box_labels = BoxLabels.load(path)

============== ============= ===========
Argument       Type          Description
============== ============= ===========
``path``       unicode       The path to load from.
**RETURNS**    ``BoxLabels`` The newly constructed object.
============== ============= ===========

----

.. image:: https://user-images.githubusercontent.com/13643239/33104476-a31678ce-cf28-11e7-993f-872f3234f4b5.png
    :alt: LightNet "logo"
