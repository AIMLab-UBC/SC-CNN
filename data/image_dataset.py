import os
import cv2
import numpy as np

import torch
import torch.utils.data

from other import utils
from other.ColorNormalization import steinseperation
import dataset.config as cfg

os.environ['KMP_DUPLICATE_LIB_OK']='True'



class CenterDataset(object):

    def __init__(self, root, patch_size, stride_size, d, out_size, version):

        self.out_size = out_size
        self.imgs = [] ; self.heat_maps = []; self.epsilons = 0

        self.transform = utils.transform(version)

        Image_path  = root + cfg.dataset_path + cfg.image_path
        Center_path = root + cfg.dataset_path + cfg.center_path

        # MacOS thing :)
        utils.delete_file(Image_path, '.DS_Store')
        utils.delete_file(Center_path, '.DS_Store')

        # load all image files, sorting them to
        total_num = len(list(os.listdir(Image_path)))

        for idx, (img_name, center_file) in enumerate(zip(list(sorted(os.listdir(Image_path))),
                                          list(sorted(os.listdir(Center_path))))):

            # Check if the img and the centers are for same file
            assert img_name[:-4]==center_file[:-4], "The Image {} and Center name {} are not same!".format(img_name, center_file)


            img_path    = os.path.join(Image_path, img_name)
            center_path = os.path.join(Center_path, center_file)

            # Load Image
            img = cv2.imread(img_path)


            # Obtaining H-Channel
            # Vahadane color normalization --> my implementation
            # of Matlab's version
            _, _, _, stain, _ = steinseperation.stainsep(img, 2, 0.02)
            H_Channel = stain[0]

            cropped, coords = utils.patch_extraction(img, H_Channel, patch_size,
                                                     stride_size, version)

            # Reading Centers
            center_txt = open(center_path, "r")
            center = utils.read_center_txt(center_txt)

            # Extracting patches' centers
            patch_centers = utils.center_extraction(center, coords, patch_size,
                                                    self.out_size)

            # Finding epsilon and heatmaps
            h_map, epsilon = utils.heat_map(patch_centers, coords, d, patch_size,
                                            self.out_size)

            self.imgs.extend(cropped)
            self.heat_maps.extend(h_map)
            self.epsilons += epsilon

            print(idx+1, 'from', total_num, 'Images are Loaded!' ,
                  sep=' ', end='\r', flush=True)


    def __getitem__(self, idx):

        img = self.imgs[idx]
        heat_map = self.heat_maps[idx]
        # Number of pixels that have a nonzero value / zero values
        epsilon = self.epsilons / (len(self.imgs)*self.out_size[0]*self.out_size[1] - self.epsilons)

        # Normalize image
        img = self.transform(np.uint8(img))
        img = img.float()
        heat_map = torch.as_tensor(heat_map, dtype=torch.float)
        epsilon = torch.as_tensor(epsilon, dtype=torch.float)

        return img, heat_map, epsilon


    def __len__(self):
        return len(self.imgs)
