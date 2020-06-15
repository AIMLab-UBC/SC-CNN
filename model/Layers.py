import torch.nn as nn


def conv_pool(in_channels, out_channels, conv_kernel_size, pool_kernel_size):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=1, padding=0),
        nn.ReLU(inplace=True),
        # nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=pool_kernel_size)
        )



def fully_connected(in_size, out_size, p):

    return nn.Sequential(
        nn.Linear(in_size, out_size),
        # nn.BatchNorm1d(out_size),
        nn.ReLU(inplace=True),
        nn.Dropout(p=p)
        )



def S1_layer(in_size, out_size):

    return nn.Sequential(
        nn.Linear(in_size, out_size),
        # nn.BatchNorm1d(out_size),
        nn.Sigmoid()
        )
