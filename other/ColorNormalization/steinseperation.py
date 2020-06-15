'''
@inproceedings{Vahadane2015ISBI,
Author = {Abhishek Vahadane and Tingying Peng and Shadi Albarqouni and Maximilian Baust and Katja Steiger and Anna Melissa Schlitter and Amit Sethi and Irene Esposito and Nassir Navab},
Booktitle = {IEEE International Symposium on Biomedical Imaging},
Date-Modified = {2015-01-31 17:49:35 +0000},
Title = {Structure-Preserved Color Normalization for Histological Images},
Year = {2015}}

This is a Python version of this paper.
Implementation of Matlab's version : https://github.com/abhishekvahadane/CodeRelease_ColorNormalization

Amirali
'''

import other.ColorNormalization.functions as functions


def stainsep(image, nstains, lambda1):

    # Image should have 3 channels --> RGB
    _, _, channels = image.shape

    if channels != 3:
        print("Image should be in RGB format ...")
        return

    V, V1 = functions.Beer_Lamber(image)

    # without float it is not working
    V, V1 = V.astype(float), V1.astype(float)

    # Flag for cheking if V1 has H or we can say that if it is not stain part, so we can remove it
    if len(V1)==0:
        stain = False
        return 0, 0, 0, 0, stain

    stain = True
    param, param2 = functions.define_params(nstains, lambda1, round(0.2 * V1.shape[0]))

    Wi = functions.get_staincolor_sparsenmf(V1, **param)

    Hi, sepstains, _ = functions.estH(V, Wi, param, param2, image.shape[0], image.shape[1])

    Hiv = Hi.reshape(-1, Hi.shape[2])

    return Wi, Hi, Hiv, sepstains, stain
