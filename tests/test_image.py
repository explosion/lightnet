from pathlib import Path
from numpy.testing import assert_equal

from lightnet.lightnet import Image

def test_make_image():
    img = Image.blank(10, 10, 10)
    img2 = Image.blank(100, 10, 10)

def test_random_image():
    img = Image.random(10, 10, 10)
    img2 = Image.random(100, 10, 10)

def test_image_from_bytes():
    path = Path("tests/COCO_val2014_000000000042.jpg")
    loaded = Image.load_color(path)
    with path.open('rb') as file_:
        raw = file_.read()
    made = Image.from_bytes(raw)
    assert_equal(made.data, loaded.data)

