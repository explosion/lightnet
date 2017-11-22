#!/usr/bin/env python
import shutil
import io
import os
import json
import distutils.command.build_ext
import subprocess
import sys
from setuptools import Extension, setup
import platform
import numpy

try:
    import cython
    use_cython = True
except ImportError:
    use_cython = False


class ExtensionBuilder(distutils.command.build_ext.build_ext):
    def build_extensions(self):
        if use_cython:
            subprocess.check_call([sys.executable, 'bin/cythonize.py'],
                                   env=os.environ)
        extensions = []
        e = self.extensions.pop(0)
        c_sources = get_c_sources(os.path.join(PWD, 'lightnet', '_darknet'))
        include_dir = os.path.join(PWD, 'lightnet', '_darknet')
        self.extensions.append(Extension(e.name, e.sources + c_sources))
        for e in self.extensions:
            e.include_dirs.append(numpy.get_include())
            e.include_dirs.append(os.path.abspath(include_dir)),
            e.undef_macros.append("FORTIFY_SOURCE")
            e.extra_compile_args.append("-DCBLAS")
            e.extra_compile_args.append("-Wno-strict-prototypes")
            if sys.platform == 'darwin':
                e.extra_compile_args.append('-D__APPLE__')
                e.extra_link_args.append('-lblas')
            else:
                e.extra_link_args.append('-lopenblas')
                e.extra_link_args.append('-L/opt/OpenBLAS/lib')
        distutils.command.build_ext.build_ext.build_extensions(self)


def get_c_sources(start_dir):
    c_sources = []
    excludes = []
    for path, subdirs, files in os.walk(start_dir):
        for exc in excludes:
            if exc in path:
                break
        else:
            for name in files:
                if name.endswith('.c'):
                    c_sources.append(os.path.join(path, name))
    return c_sources


PWD = os.path.join(os.path.dirname(__file__))
INCLUDE = os.path.join(PWD, 'lightnet', '_darknet')

c_files = get_c_sources(os.path.join(PWD, 'lightnet', '_darknet'))

with io.open(os.path.join(PWD, 'lightnet', 'about.py'), encoding='utf8') as f:
    about = {}
    exec(f.read(), about)

with io.open(os.path.join(PWD, 'README.rst'), encoding='utf8') as f:
    readme = f.read()

setup(
    setup_requires=['numpy'],
    install_requires=['numpy', 'plac', 'requests', 'pathlib', 'tqdm'],
    ext_modules=[
        Extension('lightnet.lightnet', ['lightnet/lightnet.c']),
    ],
    cmdclass={'build_ext': ExtensionBuilder},
    package_data={'': ['*.json', '*.pyx', '*.pxd', '_darknet/*.h',
                       'data/*.cfg', 'data/*.template', 'data/*.names'] + c_files},

    name=about['__title__'],
    zip_safe=False,
    packages=['lightnet'],
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__email__'],
    url=about['__uri__'],
    license=about['__license__'],
    description=about['__summary__'],
    long_description=readme,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
)
