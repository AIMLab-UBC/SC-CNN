import os
import numpy as np
import torch
import matplotlib.pyplot as plt



def printInformation(dataset, idx):

    '''
    This function shows the image, heatmap, and epsilon for each data by
    the index.

    Note: If you see the images, it is not as you expected since I have used
          color normalization.

    Input:
          1- Dataset:
                The dataset that the images, .. are in that.
          2- idx:
                Index of which images we want to get the information.
    '''

    img, heat_map, epsilon = dataset[idx]

    print(5*'*')
    print('Heat Map:', heat_map)
    print(5*'*')
    print('epsilon:', epsilon)

    # Show Image
    plt.imshow(img.permute(1, 2, 0))
    plt.title('Image')
    plt.show()



def patch_extraction(img, patch, stride):

    '''
    This function extracts patches from images.

    Input:
          1- img:
                The img we want to extract patches from that.
          2- patch:
                Size of patches. It should be in this format: [p_h, p_w]
          3- stride:
                Size of stride. It should be in this format: [s_h, s_w]

    Output:
          1- samples:
                List of patches.
          2- coords:
                coordinates of patches in original image.
    '''

    # gray --> change it to have a channel
    if len(img.shape) < 3:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    img_H, img_W, img_C = img.shape[0], img.shape[1], img.shape[2]
    patch_H, patch_W = patch[0], patch[1]
    stride_H, stride_W = stride[0], stride[1]

    num_patch_H = (img_H - patch_H)/stride_H + 1
    num_patch_W = (img_W - patch_W)/stride_W + 1

    if not isinstance(num_patch_H, int):
        padd_H = int(num_patch_H) * stride_H + patch_H - img_H
        num_patch_H = int(num_patch_H) + 1
    else:
        padd_H = 0

    if not isinstance(num_patch_W, int):
        padd_W = int(num_patch_W) * stride_W + patch_W - img_W
        num_patch_W = int(num_patch_W) + 1
    else:
        padd_W = 0

    padded_img = np.zeros([img_H+padd_H, img_W+padd_W, img_C])
    padded_img[ :img_H, :img_W, :] = img

    samples = list()
    coords = list()

    for i in range(num_patch_H):
        for j in range(num_patch_W):

            start_x = i*stride_H ; end_x = start_x + patch_H
            start_y = j*stride_W ; end_y = start_y + patch_W

            crop_image = padded_img[ start_x : end_x, start_y : end_y, :]

            samples.append(crop_image.astype(int))
            coords.append(np.array([[start_x, start_y],
                                    [end_x, end_y]]))

    return samples, coords


def read_center_txt(text_file):

    '''
    This function reads the centers from .txt files.

    Input:
          1- text_file:
                Path to the .txt file.

    Output:
          1- centers:
                Coordinates of centers.
    '''

    lines = text_file.read().split('\n')
    lines = lines[:-1]
    centers = np.zeros((len(lines), 2))

    for idx, line in enumerate(lines):

        numbers = line[1:-1].split(',')
        x = int(numbers[0]) ; y = int(numbers[1])
        centers[idx][0] = x ; centers[idx][1] = y

    return centers


def find_out_coords(coord, patch_size, out_size):

    '''
    This function determines the coordinates of output's window. This window is
    in the center of the patch.

    Input:
          1- coord:
                Coordinates of the Patch.It should be in this format: [[start_x, start_y]
                                                                       [end_x  , end_y  ]]
          2- patch_size:
                Size of patches. It should be in this format: [p_h, p_w]
          3- out_size:
                Size of output map. It should be in this format: [o_h, o_w]

    Output:
          1- centers:
                Coordinates of centers.
    '''

    H_prime, W_prime = out_size[0]  , out_size[1]
    patch_H, patch_W = patch_size[0], patch_size[1]

    ## If want to consider all
    # start_H = coords[i][0][0]; start_W = coords[i][0][1]
    # end_H   = coords[i][1][0]; end_W   = coords[i][1][1]

    ## If just the out_size
    margin_H = (patch_H - H_prime)/2      ; margin_W = (patch_W - W_prime)/2
    start_H  = int(coord[0][0] + margin_H); start_W  = int(coord[0][1] + margin_W)
    end_H    = start_H + H_prime          ; end_W    = start_W + W_prime

    assert end_H < coord[1][0] or end_W < coord[1][1], "H --> End: {} Coord: {} W --> End: {} Coord: {}".format(end_H, coord[1][0], end_W, coord[1][1])
    assert start_H > coord[0][0] or start_W > coord[0][1], "H --> Start: {} Coord: {} W --> Start: {} Coord: {}".format(start_H, coord[0][0], start_W, coord[0][1])

    return start_H, end_H, start_W, end_W


def center_extraction(centers, coords, patch_size, out_size):

    '''
    This function extracts the centers which place in the output window of
    the patch. For example, if the patch size is [27*27] and the output size is
    [11, 11], this functions extracts the centers that are in the [11*11] window
    of the patch.

    Input:
          1- coord:
                Coordinates of the Patch.It should be in this format: [[start_x, start_y]
                                                                       [end_x  , end_y  ]]
          2- patch_size:
                Size of patches. It should be in this format: [p_h, p_w]
          3- out_size:
                Size of output map. It should be in this format: [o_h, o_w]

          Note: This formats are for one data. The input of this function is
                list of these points.

    Output:
          1- centers:
                Coordinates of centers.
    '''

    patch_centers = list()

    for i in range(len(coords)):

        start_H, end_H, start_W, end_W = find_out_coords(coords[i], patch_size,
                                                         out_size)

        patch_center =  [point for point in centers if point[0] > start_H and
                         point[0] < end_H and point[1] > start_W and point[1] < end_W]

        patch_centers.append(patch_center)

    return patch_centers



def euclidean_dist_squared(X, Xtest):

    '''
    This function Computes the Euclidean distance between rows of 'X' and rows
    of 'Xtest'.

    Credit: CPSC340 HW
    '''

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)


def heat_map(patch_centers, coords, d, patch_size, out_size):

    '''
    This function calculates the heatmap.

    Input:
          1- patch_centers:
                Coordinates of the centers of patch that are in the output window.
          2- coords:
                Coordinates of points from original image where patches are
                extracted.
          3- d:
                Distance from center that below that has a value in heatmap.
          4- patch_size:
                Size of patches. It should be in this format: [p_h, p_w]
          5- out_size:
                Size of output map. It should be in this format: [o_h, o_w]

    Output:
          1- map:
                List of heatmaps.
          2- epsilon:
                List of epsilons.

    Note: The epsilon is written in this way, so if I want to have different
          value for each patch, I can easily change here.
    '''

    maps = list()

    # epsilon = list()
    epsilon = 0

    H_prime, W_prime = out_size[0]  , out_size[1]

    for idx, coord in enumerate(coords):

        start_H, end_H, start_W, end_W = find_out_coords(coord, patch_size,
                                                         out_size)
        if len(patch_centers[idx]) == 0:
            out = np.zeros((H_prime, W_prime))
            eps = 0

        else:

            Xtrain = [[a, b] for a in range(start_H, end_H) for b in range(start_W, end_W)]
            Xtrain = np.array(Xtrain).astype(int)

            Xtest = np.zeros((len(patch_centers[idx]), 2))

            for i, point in enumerate(patch_centers[idx]):
                Xtest[i, :] = point

            Xtest = Xtest.astype(int)
            distance = euclidean_dist_squared(Xtrain, Xtest)

            out = np.min(distance, axis=1)
            out = np.minimum(out, d**2)

            out[out==d**2]=0
            out = out.reshape(H_prime, W_prime)

            out = 1./(1+out/2)
            out[out==1]=0

            # Centers should be 1
            for point in Xtest:
                out[point[0]-start_H][point[1]-start_W]=1

            eps = len(np.nonzero(out)[0])

        maps.append(out)
        epsilon += eps

    return maps, epsilon


def euclidean_dist_squared_tensor(data, center):

    '''
    This function calculates the heatmap.

    Input:
          1- patch_centers:
                Coordinates of the centers of patch that are in the output window.
          2- coords:
                Coordinates of points from original image where patches are
                extracted.
          3- d:
                Distance from center that below that has a value in heatmap.
          4- patch_size:
                Size of patches. It should be in this format: [p_h, p_w]
          5- out_size:
                Size of output map. It should be in this format: [o_h, o_w]

    Output:
          1- map:
                List of heatmaps.
          2- epsilon:
                List of epsilons.

    Note: The epsilon is written in this way, so if I want to have different
          value for each patch, I can easily change here.
    '''

    n = data.size(0)
    m = center.size(0)
    d = data.size(1)

    data = data.unsqueeze(1).expand(n, m, d)
    center = center.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(data - center, 2).sum(2)

    return dist


def heat_map_tensor(points, h_value, device, d, out_size):

    '''
    This function calculates the heatmap for tensors.

    Input:
          1- points:
                X and Y for the center of cell.
          2- h_value:
                Coefficient for heatmap values.
          3- device:
                For Using Multiple GPUs.
          4- d:
                Distance from center that below that has a value in heatmap.
          5- out_size:
                Size of output map. It should be in this format: [o_h, o_w]

    Output:
          1- map:
                List of heatmaps.
          2- epsilon:
                List of epsilons.

    '''

    H = out_size[0]; W = out_size[1]

    temp = torch.tensor([[h, w] for h in range(H) for w in range(W)], device=device)

    batch_size = points.shape[0]

    result = torch.zeros((batch_size, H, W), device=device)

    for i in range(batch_size):

        center = points[i,:]

        dist = euclidean_dist_squared_tensor(temp, center)

        min_val, _ = torch.min(dist, dim=1, keepdim=True)
        max_dist   = torch.clamp(min_val, max=d**2)

        max_dist[max_dist==torch.max(max_dist)] = 0

        out = 1./(1+max_dist/2)
        out[out==1] = 0
        out = out.view(H, W)
        idx = center.detach()
        idx = idx.int()
        out[idx[0], idx[1]] = 1

        result[i, :] = out * h_value[i]

    return result


def heat_map_tensor_version2(points, h_values, device, d, out_size):

    '''
    This function calculates the heatmap for tensors.

    Note: The output is same as previous function, but in a different way to
          compare the gradient.
          This function because of 'for loops' is not efficient!

    '''

    H = out_size[0]; W = out_size[1]

    batch_size = points.shape[0]

    result = torch.zeros((batch_size, H, W), device=device)

    for i in range(batch_size):

        center = points[i,:]
        h_value = h_values[i]

        for h in range(H):
            for w in range(W):

                if torch.pow(h - center[0], 2) + torch.pow(w - center[1], 2) <= d**2:

                    result[i, h, w] = h_value / (1 + (torch.pow(h - center[0], 2) + torch.pow(w - center[1], 2))/2)

    return result


def delete_file(path, name):

    '''
    This function delets the unwanted files in data folder. This is for MacOS
    that in the datafolder there is '.Ds_Store' file.
    '''

    for file in list(sorted(os.listdir(path))):
        if file == name:
            os.remove(path + '/' + name)
