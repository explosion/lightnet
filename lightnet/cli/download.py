# coding: utf8
from __future__ import unicode_literals

import plac
import requests
import os
import sys
from tqdm import tqdm
from pathlib import Path


model_paths = {
    'yolo': 'https://pjreddie.com/media/files/yolo.weights',
    'tiny-yolo': 'https://pjreddie.com/media/files/tiny-yolo.weights',
}

@plac.annotations(
    model=("model to download, shortcut or name)", "positional", None, str),
    direct=("force direct download. Needs model name with version and won't "
            "perform compatibility check", "flag", "d", bool))
def download(cmd, model, direct=False):
    """
    Download model from default download path. Models: tiny-yolo, yolo, yolo2
    """
    if direct:
        url = model
        name = model.split('/')[-1]
    else:
        url = model_paths[model]
        name = model + '.weights'
    out_loc = Path(__file__).parent.parent / 'data' / name
    download_file(url, out_loc)


def download_file(url, path):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    with Path(path).open('wb') as file_:
        with tqdm(total=total_size//1024, unit_scale=True, unit="K") as pbar:
            for data in r.iter_content(32*1024):
                file_.write(data)
                pbar.update(32)
