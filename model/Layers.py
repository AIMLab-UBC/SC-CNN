import torch.nn as nn


def conv_pool(in_channels, out_channels, conv_kernel_size, pool_flag=True,
              pool_kernel_size=2):

    layers = []

    layers.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=conv_kernel_size, stride=1, padding=0))
    layers.append(nn.ReLU(inplace=True))

    if pool_flag:
        layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size))

    return nn.Sequential(*layers)



def fully_conv(in_channels, out_channels, conv_kernel_size, p):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size,
                  stride=1, padding=0),
        # nn.BatchNorm1d(out_size),
        nn.ReLU(inplace=True),
        nn.Dropout(p=p)
        )



def fully_connected(in_size, out_size, p):

    return nn.Sequential(
        nn.Linear(in_size, out_size),
        # nn.BatchNorm1d(out_size),
        nn.ReLU(inplace=True),
        nn.Dropout(p=p)
        )


def S1_layer_fc(in_size, out_size):

    return nn.Sequential(
        nn.Linear(in_size, out_size),
        # nn.BatchNorm1d(out_size),
        nn.Sigmoid()
        )

def S1_layer_conv(in_size, out_size):

    return nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
        # nn.BatchNorm1d(out_size),
        nn.Sigmoid()
        )
