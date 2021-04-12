import ast
import base64
from io import BytesIO
import os
import json

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image


def json2img(j):
    try:
        data = json.loads(j)
    except:
        data = ast.literal_eval(j)
    if type(data) is str:
        data = ast.literal_eval(data)
    img = Image.open(BytesIO(base64.b64decode(data['screen']['data'])))
    with open('data.json', 'w') as f:
        json.dump(data, f)
    img_array = np.array(img)
    return img_array


def read_image_csv(path, n_images=None, load_states=False):
    data = []
    states = []
    filenames = np.arange(1)
    for f in filenames:
        print(f)
        df = pd.read_csv(path, error_bad_lines=False)
        df_screen = df[df["Command"] == "SENSE_SCREEN"]
        if n_images is None:
            n_images = len(df_screen["JSON"])
        for i, j in enumerate(df_screen["JSON"]):
            if i == n_images:
                break
            if "screen" in j:
                with open('data.json', 'w') as f:
                    json.dump(j, f)
                data.append(json2img(j))
        if load_states:
            df_senseall = df[df["Command"] == "SENSE_ALL"]
            for i, j in enumerate(df_senseall["JSON"]):
                if i == n_images:
                    break
                states.append(json.loads(j))
    data = np.array(data)
    imsize1 = data.shape[1]
    imsize2 = data.shape[2]
    channels = data.shape[-1]
    data = data.reshape(-1, imsize1, imsize2, channels)
    return data, states


def read_images(path):
    from os import walk
    f = []
    data = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
    for f in filenames:
        ima1 = cv.imread(os.path.join(path, f))
        ima1 = cv.resize(ima1, (64, 64))/255
        data.append(ima1)
    data = np.array(data)
    imsize1 = data.shape[1]
    imsize2 = data.shape[2]
    channels = data.shape[-1]
    data = data.reshape(-1, imsize1, imsize2, channels)
    return data, filenames


def threshold_image_mask(img, th):
    img /= np.max(img)
    img[np.where(img < th)] = 0
    img[np.where(img > th)] = 1
    return img
