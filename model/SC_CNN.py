import torch
import torch.nn as nn

from model.Layers import conv_pool, fully_connected, S1_layer


class SC_CNN(nn.Module):

    '''
    This class is SC-CNN layers based on original paper.

    The class has 2 outputs:

        1- points (centers)
        2- H

    Note: In original paper, the points are obtained from:
            x = (Height-1)*Sigmoid(..) + 1
            y = (Width -1)*Sigmoid(..) + 1

          Since it is in Matlab and Matlab is 1-based index, but Python is
          0-based index, so in here it is:

            x = (Height-1)*Sigmoid(..)
            y = (Width -1)*Sigmoid(..)

    Amirali
    '''

    def __init__(self, M, out_size):
        super().__init__()

        self.conv_1 = conv_pool(3, 36, conv_kernel_size=4, pool_kernel_size=2)
        self.conv_2 = conv_pool(36, 48, conv_kernel_size=3, pool_kernel_size=2)

        self.fc_1 = fully_connected(5*5*48, 512, 0.2)
        self.fc_2 = fully_connected(512, 512, 0.2)

        self.S1_point = S1_layer(512, 2*M)
        self.S1_h = S1_layer(512, M)

        H_prime, W_prime = out_size[0], out_size[1]
        self.border_1 = torch.tensor([H_prime-1, W_prime-1])

        ## Initialization
        # Bias
        torch.nn.init.constant_(self.conv_1[0].bias, 0)
        torch.nn.init.constant_(self.conv_2[0].bias, 0)
        torch.nn.init.constant_(self.fc_1[0].bias, 0)
        torch.nn.init.constant_(self.fc_2[0].bias, 0)
        torch.nn.init.constant_(self.S1_point[0].bias, 0)
        torch.nn.init.constant_(self.S1_h[0].bias, 0)

        # Weights
        torch.nn.init.normal_(self.conv_1[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.conv_2[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc_1[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc_2[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.S1_point[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.S1_h[0].weight, mean=0.0, std=0.01)


    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)

        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)

        points = self.S1_point(x)

        # For Using Multiple GPUs, it is necessary
        if torch.cuda.is_available():
            points = points * self.border_1.to(x.get_device())
        else:
            points = points * self.border_1

        h = self.S1_h(x)

        return points, h
