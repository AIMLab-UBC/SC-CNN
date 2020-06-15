'''
This is the implementation of following paper:

    @inproceedings{Vahadane2015ISBI,
    Author = {Korsuk Sirinukunwattana and Shan E Ahmed Raza and Yee-Wah Tsang and David R. J. Snead and Ian A. Cree and Nasir M. Rajpoot},
    Booktitle = {IEEE TRANSACTIONS ON MEDICAL IMAGING},
    Title = {Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images},
    Year = {2016}}


Amirali Darbandsari
AIM Lab
'''


import os
from other.parser import parse_input
from dataset.dataset import dataset
from train import train


if __name__ == "__main__":

    arg = parse_input()

    if arg.mode == 'train':

        # Path to this file
        # -7 is the lenght of the name of the file 'main.py'
        # to get the folder path
        file_path = os.path.realpath(__file__)
        root = file_path[:-7]

        # Download the dataset
        dataset(root)

        # Training Phase
        train(arg)
