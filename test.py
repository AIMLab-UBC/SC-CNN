import os
import torch
import numpy as np
from PIL import Image
from numpy import asarray
import other.utils as utils
from model.SC_CNN import SC_CNN
from model.SC_CNN_v2 import SC_CNN_v2
from other.ColorNormalization import steinseperation




def print_values(img, model, arg):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img_transform = utils.transform(arg.version)
    img = img_transform(np.uint8(img))
    img = img.float()

    img = img.to(device)

    img = img.unsqueeze(0)
    model.train(False)

    point, h = model(img)
    print('points     :', point.cpu().detach().numpy().reshape((-1, 2)))
    print('H          :', h.cpu().detach().numpy().reshape((-1, 1)))
    print(20*'*')

    return utils.heat_map_tensor(point.view(-1, 2), h.view(-1, 1),
                                 device, arg.d, arg.heatmap_size)



def test(arg):

    root = os.path.realpath(__file__)[:-7]

    if arg.version==0 or arg.version==1:
        # get the original model
        model = SC_CNN(arg.M, arg.patch_size, arg.heatmap_size, arg.version)

    if arg.version==2:
        # get the new model
        model = SC_CNN_v2(arg.M, arg.heatmap_size)


    # load model
    _, _, model, _, _ = utils.load_model(arg.load_flag, arg.load_name, model,
                                         None, None)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    image = Image.open(os.path.join(root + '/Datasets/CRCHistoPhenotypes_2016_04_28/Tissue Images/img1.png'))
    data = asarray(image)

    _, _, _, stain, _ = steinseperation.stainsep(data, 2, 0.02)
    H_Channel = stain[0]

    cropped, coords = utils.patch_extraction(data, H_Channel, arg.patch_size,
                                             arg.stride_size, arg.version)

    H, W, _ = data.shape
    cell = np.zeros((H, W))
    count = np.zeros((H, W))

    [H_prime, W_prime] = arg.heatmap_size

    for img, coord in zip(cropped, coords):

        heat_map = print_values(img, model, arg)
        heatmap  = heat_map.cpu().detach().numpy().reshape((H_prime, W_prime))

        start_H, end_H, start_W, end_W = utils.find_out_coords(coord, arg.patch_size,
                                                               arg.heatmap_size)

        cell[start_H:end_H, start_W:end_W] += heatmap

        idx = np.argwhere(heatmap != 0)
        count[idx[:,0]+start_H, idx[:,1]+start_W] += 1

    count[count==0] = 1
    cell = np.divide(cell, count)

    return cell
