import torch
import torch.nn as nn

from model.Layers import conv_pool, fully_conv, S1_layer_conv


class SC_CNN_v2(nn.Module):

    '''
    This class is SC-CNN layers based on weights provided by Authors.

    The class has 2 outputs:

        1- points (centers)
        2- H

    Amirali
    '''

    def __init__(self, M, out_size):
        super().__init__()

        self.conv_1 = conv_pool(4, 30, conv_kernel_size=2, pool_flag=True,
                                pool_kernel_size=2)
        self.conv_2 = conv_pool(30, 60, conv_kernel_size=2, pool_flag=True,
                                pool_kernel_size=2)
        self.conv_3 = conv_pool(60, 90, conv_kernel_size=3, pool_flag=False,
                                pool_kernel_size=2)


        self.full_conv_1 = fully_conv(90, 1024, conv_kernel_size=5, p=0.5)
        self.full_conv_2 = fully_conv(1024, 512, conv_kernel_size=1, p=0.5)


        self.S1_point = S1_layer_conv(512, 2*M)
        self.S1_h = S1_layer_conv(512, M)

        H_prime, W_prime = out_size[0], out_size[1]
        # View in point shape
        self.border_1 = torch.tensor([H_prime-1, W_prime-1]).view(1,2,1,1)

        ## Initialization
        # Bias
        torch.nn.init.constant_(self.conv_1[0].bias, 0.0)
        torch.nn.init.constant_(self.conv_2[0].bias, 0.0)
        torch.nn.init.constant_(self.conv_3[0].bias, 0.0)
        torch.nn.init.constant_(self.full_conv_1[0].bias, 0.0)
        torch.nn.init.constant_(self.full_conv_2[0].bias, 0.0)
        torch.nn.init.constant_(self.S1_point[0].bias, 0.0)
        torch.nn.init.constant_(self.S1_h[0].bias, 0.0)

        # Weights
        torch.nn.init.normal_(self.conv_1[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.conv_2[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.conv_3[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.full_conv_1[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.full_conv_2[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.S1_point[0].weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.S1_h[0].weight, mean=0.0, std=0.01)


    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        x = self.full_conv_1(x)
        x = self.full_conv_2(x)

        points = self.S1_point(x)
        # For Using Multiple GPUs, it is necessary
        if torch.cuda.is_available():
            points = points * self.border_1.to(x.get_device())
        else:
            points = points * self.border_1

        h = self.S1_h(x)

        return points, h
