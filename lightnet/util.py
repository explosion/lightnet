from contextlib import contextmanager
from tempfile import mkdtemp
import shutil


@contextmanager
def make_temp_dir():
    path = mkdtemp()
    yield path
    shutil.rmtree(path)
