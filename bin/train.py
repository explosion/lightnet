from lightnet.lightnet import train
import plac
from pathlib import Path

try:
    unicode
except NameError:
    unicode = str

def path2bytes(loc):
    return unicode(Path(loc).resolve()).encode('utf8')

def main(cfg_loc, weight_loc, images_loc):
    train(path2bytes(cfg_loc), path2bytes(weight_loc),
          path2bytes(images_loc), path2bytes('/tmp/yolo'))

if __name__ == '__main__':
    plac.call(main)
