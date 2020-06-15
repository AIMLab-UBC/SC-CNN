import os
from os import path
from scipy.io import loadmat
from shutil import copy
import requests, zipfile, io


def save_center(annotation_file, name, center_path):

        '''
        This function reads the .mat files and write the Centers
        in .txt file.
        '''

        annots = loadmat(annotation_file)

        annots['detection'] = annots['detection'].astype(int)

        with open(center_path + name + '.txt', 'w') as f:
            for point in annots['detection']:
                # Because it is written in Matlab
                # and Matlab start from {1} instead of {0}
                f.write("{}\n".format((point[0]-1, point[1]-1)))


def load_dataset(main_path, image_path, center_path):

    # if not, create it
    if not os.path.isdir(image_path):
        os.mkdir(image_path)
        # print("Image Folder Created .....")

    if not os.path.isdir(center_path):
        os.mkdir(center_path)
        # print("Center Folder Created .....")

    for root, dirs, files in os.walk(main_path, topdown=True):
        for name in sorted(files):

            if name == '.DS_Store':
                continue

            if name.endswith('.bmp'):
                copy(os.path.join(root, name), image_path)
                os.rename(os.path.join(image_path, name), os.path.join(image_path, name[:-4]+'.png'))

            if name.endswith('.mat'):
                save_center(os.path.join(root, name), name[:-14], center_path)


def download_dataset(url, path):

    '''
    This function downloads the dataset and extracts the .zip file.

    Input:
          1- url:
                The link to the dataset.
          2- path:
                The path we want to extract the file in it.
    '''

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
