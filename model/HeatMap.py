import torch
import other.utils as utils


class HeatMap(torch.autograd.Function):

    '''
    This class calculates the Heatmap from the points and h that are outputs of
    previous model.

    Note: For now, it is just implemented for M=1!

    Amirali
    '''

    @staticmethod
    def forward(ctx, points, h_value, device, d, out_size):

        heat_map = utils.heat_map_tensor(points, h_value, device, d, out_size)
        ctx.save_for_backward(points, h_value, heat_map)

        ctx.out_size = out_size
        ctx.device = device

        return heat_map


    @staticmethod
    def backward(ctx, grad_output):

        out_size, device = ctx.out_size, ctx.device
        points, h_value, heat_map = ctx.saved_tensors

        batch_size = heat_map.shape[0]
        H, W = out_size[0], out_size[1]
        temp = torch.pow(heat_map, 2) / h_value.view(batch_size, 1, 1)
        
        # replace Nan with zero
        temp[temp != temp] = 0
        
        coords = torch.tensor([[h, w] for h in range(H) for w in range(W)], device=device)

        x = coords[:, 0]; y = coords[:, 1]
        x, y = x.reshape(H, W), y.reshape(H, W)

        x_points, y_points = points[:, 0], points[:, 1]

        num_x = x.view(1, H, W) - x_points.view(batch_size, 1, 1)
        num_y = y.view(1, H, W) - y_points.view(batch_size, 1, 1)

        grad_x = num_x * temp
        grad_y = num_y * temp

        grad_x = torch.sum(grad_x*grad_output, (1,2))
        grad_y = torch.sum(grad_y*grad_output, (1,2))

        grad_points = torch.stack([grad_x, grad_y], dim=1)

        grad_h_value = heat_map / h_value.view(heat_map.shape[0], 1, 1)
        grad_h_value = torch.sum(grad_h_value*grad_output, (1,2)).reshape(batch_size, 1)


        return grad_points, grad_h_value, None, None, None
