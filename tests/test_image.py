from lightnet.lightnet import Image

def test_make_image():
    img = Image(10, 10, 10)
    img2 = Image(100, 10, 10)

def test_random_image():
    img = Image.random(10, 10, 10)
    img2 = Image.random(100, 10, 10)
