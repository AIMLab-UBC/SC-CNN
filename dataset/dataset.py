import os
import dataset.config as cfg
from dataset.functions import load_dataset, download_dataset


def dataset(folder):

    '''
    This function checks if the dataset is not downloaded previously, it
    downloads that.

    Input:
          1- folder:
                The absolute path to the main folder.
    '''

    path = os.path.join(folder, cfg.dataset_path)

    # If the data is not downloaded yet
    if not os.path.isdir(path):

        print(20 * '*')
        print('Downloading the Dataset:')
        print()

        os.mkdir(path)

        download_dataset(cfg.url, path)
        load_dataset(path + cfg.detection_path, path + cfg.image_path, path + cfg.center_path)

    else:

        print(20 * '*')
        print('Dataset Exists!')
        print()
