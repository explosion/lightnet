# coding: utf8
from __future__ import unicode_literals

import plac
import requests
import os
import sys


model_paths = {
    'yolo': 'https://pjreddie.com/media/files/yolo.weights',
    'yolo2': 'https://pjreddie.com/media/files/yolo.weights',
    'tiny-yolo': 'https://pjreddie.com/media/files/tiny-yolo.weights',
}

@plac.annotations(
    model=("model to download, shortcut or name)", "positional", None, str),
    direct=("force direct download. Needs model name with version and won't "
            "perform compatibility check", "flag", "d", bool))
def download(cmd, model):
    """
    Download model from default download path. Models: tiny-yolo, yolo, yolo2
    """
    if direct:
        path = model
    else:
        path = model_paths[model]
    dl = download_model(path)
    r = requests.get(url)
    if r.status_code != 200:
        msg = ("Couldn't fetch %s. Please find a model for your spaCy "
               "installation (v%s), and download it manually.")
        prints(msg % (desc, about.__version__), about.__docs_models__,
               title="Server error (%d)" % r.status_code, exits=1)
