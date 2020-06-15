import torch


class BCE_Loss(torch.autograd.Function):

    '''
    This class is Binary Cross Entropy with Weighting that is used in the paper
    In paper, the loss is defined:

        Loss =  - (y+epsilon) * (ylog(x) - (1-y)log(1-x))

    I used Pytorch's idea for when x = 0 in log(x), where I used log-clamp.
    if log(x) is below -100, the value of log is -100
    and the gradient is 0.

    Amirali
    '''

    @staticmethod
    def forward(ctx, x, y, weight):

        ctx.save_for_backward(x, y, weight)

        clamp_log_x = torch.log(x)
        clamp_log_x[clamp_log_x <-100] = -100

        clamp_log_1_x = torch.log(1-x)
        clamp_log_1_x[clamp_log_1_x <-100] = -100

        loss = -y * clamp_log_x + (1-y) * clamp_log_1_x
        loss = loss * weight

        return loss


    @staticmethod
    def backward(ctx, grad_output):

        x, y, weight = ctx.saved_tensors

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        temp_x = 1 / x
        temp_x[torch.log(x)<-100] = 0

        temp_1_x = 1 / (1-x)
        temp_1_x[torch.log(1-x)<-100] = 0

        dloss_dx = -y * temp_x - (1-y) * temp_1_x
        dloss_dx = weight * dloss_dx

        return dloss_dx, None, None
