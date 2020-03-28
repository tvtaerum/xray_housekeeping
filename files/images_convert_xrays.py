# example of extracting and resizing xrays into a new dataset
from os import listdir
from numpy import asarray
from numpy import savez_compressed
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pandas as pd
import numpy as np

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
varNames = ["NORMAL","BACTERIA","VIRUS"]
varNames = ["NORMAL","PNEUMONIA"]

# load an image as an rgb numpy array
def load_image(filename):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    return pixels
 
# extract the xray from a loaded image and resize
def extract_xray(pixels, required_size=(80, 80)):
    # resize pixels to the model size
    image = Image.fromarray(pixels)
    image = image.resize(required_size)
    xray_array = asarray(image)
    return xray_array
 
# load images and extract xrays for all images in a directory
def load_xrays(directory, n_xrays):
    xrays = list()
    ids = list()
    labels = list()
    for dirname in listdir(directory):
        directory1=directory+dirname
        print("directory1: ", directory1)
        for dirname1 in listdir(directory1):
            directory2=directory1+"/"+dirname1
            print("directory2: ", directory2)
            # enumerate files
            for idx, filename in enumerate(listdir(directory2)):
                # load the image
                if "virus" in filename: label=1
                elif "bacteria" in filename: label=2
                else: label=0
                pixels = load_image(directory2 + "/" + filename)
                # print("pixels.size: ", pixels.size)
                # get xray
                xray = extract_xray(pixels)
                # print("pixels.size after extract: ", pixels.size)
                # if xray is None:
                    # continue
                # if data_attractive[idx] == -1.0:
                    # continue
                # store
                xrays.append(xray)
                ids.append(idx)
                labels.append(label)
                if (len(xrays)+1)%100==0:
                    print(len(xrays)+1, xray.shape)
                # stop once we have enough
                if len(xrays) >= n_xrays:
                    break
    return asarray(xrays),asarray(labels)
 
# directory that contains all images
directory = 'xray/chest_xray/'
# load and extract all xrays
n_xrays = 50000
# n_xrays = 100
all_xrays, all_labels = load_xrays(directory, n_xrays)
print('Loaded xrays: ', all_xrays.shape)
print('Loaded labels: ', all_labels.shape)
qSave = True
if qSave:
    savez_compressed('xray/img_align_xray.npz', all_xrays)
    savez_compressed('xray/labels_align_xray.npz', all_labels)


