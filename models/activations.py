"""
A zoo of fancy activation functions.
"""
import torch
from torch import nn
from torch.nn.init import _calculate_correct_fan
import math
from typing import Callable


class SymSum(nn.Module):
    "symmeric activation"
    def __init__(self):
        super(SymSum, self).__init__()

    def forward(self, x):
        zero = torch.tensor(0.).to(x.device)
        shift = x.shape[1] // 2  # This is so that R+(a) + R-(b) and R-(a) + R+(b)
        relu = torch.maximum(zero, x)
        inv_relu = torch.minimum(zero, x)
        out = relu + torch.roll(inv_relu, shift, dims=1)
        return out


class SoftSymSum(nn.Module):
    """soft symmeric activation"""
    def __init__(self, beta=3, threshold=20):
        super(SoftSymSum, self).__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x):
        shift = x.shape[1] // 2  # This is so that R+(a) + R-(b) and R-(a) + R+(b)
        soft_relu = self.softplus(x)
        inv_soft_relu = self.softplus(-x)
        out = soft_relu - torch.roll(inv_soft_relu, shift, dims=1)
        return out


class SoftShrinkHardTanh(nn.Module):
    """Corresponds to SSHT in the paper."""
    def __init__(self, softshrink_coef=1.):
        super(SoftShrinkHardTanh, self).__init__()
        self.softshrink = torch.nn.Softshrink(lambd=1)
        self.hardtanh = torch.nn.Hardtanh()
        self.softshrink_coef = torch.tensor(softshrink_coef)

    def forward(self, x):
        shift = x.shape[1] // 2  # This is so that R+(a) + R-(b) and R-(a) + R+(b)
        x1 = self.softshrink(x)
        if self.softshrink_coef != 1.:
            self.softshrink_coef.to(x.device)
            x1 = x1*self.softshrink_coef
        x2 = self.hardtanh(x)
        out = torch.roll(x1, shift, dims=1) + x2
        return out


class LazyPReLU(nn.Module):
    """
    PReLU that initializes the number of channels automatically on the first forward pass.
    The slope is initialized to 1 to compare with our symlinear initialization method.
    """
    def __init__(self, init_neg_slope=1.):
        super(LazyPReLU, self).__init__()
        self.prelu = None
        self.init_neg_slope = init_neg_slope

    def forward(self, x):
        if self.prelu is None:
            with torch.no_grad():
                self.prelu = nn.PReLU(num_parameters=x.shape[1], init=self.init_neg_slope).to(x.device)
        out = self.prelu(x)
        return out


def make_initializer(module: Callable, **kwargs):
    def init_module():
        return module(**kwargs)
    return init_module


# An ordinary implementation of Swish function
class Swish(nn.Module):
    @staticmethod
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def calculate_gain(nonlinearity: str, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity in ['relu', 'swish']:
        # Note: For swish, efficientNet seems to use the same initialization as ReLU.
        #  Ideally it should be different, but this is probability close enough.
        return math.sqrt(2.0)
    elif nonlinearity in ['leaky_relu', 'leakyrelu']:
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    elif nonlinearity in ['symsum', 'softsymsum']:
        # Todo: To be eventually replaced with the theoretically derived value.
        return param
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def correlated_kaiming_normal(tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu', corr_coef: float = 0.):
    """
    Does Kaiming normal initialization with the option to make complementary weights (when SymSum is used) are positively correlated.
    It does so by initializing the weights normally (like the default kaiming normal would do), then it adds a common noise to complementary weights.
    Note: It onlu works for 2D convolution layers for now.
    Todo: Not sure it is valid for grouped convolutions.
    corr_coef=0 would make this funnction identical to the standard kaiming_normal.
    """
    assert 0 <= corr_coef <= 1
    fan = _calculate_correct_fan(tensor, mode)
    a = 0.2 if nonlinearity == 'leakyrelu' else a  # Note: currently 0.2 is hardcoded in `get_act_func`
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        tensor.normal_(0, std)  # this is the standard uncorrelated Kaiming He initialization
        sh = tensor.shape
        if corr_coef > 0 and (sh[1] % 2 == 0):
            # The `sh[1] % 2 == 0` condition makes sure we skip layers which have odd number of input channels.
            common_noise = torch.normal(0, std, size=(sh[0], sh[1]//2, sh[2], sh[3]))
            common_noise = torch.cat([common_noise]*2, dim=1)
            # The `tensor` and the `common_noise` are scaled such that their sum has the same standard deviation.
            #  At the same time their Pearson Corr. Coef. will be `corr_coef`
            tensor.copy_((common_noise * corr_coef**0.5 + tensor * (1 - corr_coef)**0.5))
    return tensor


def get_act_func(activation: str) -> Callable:
    if activation == 'relu':
        act_func = nn.ReLU
    elif activation == 'prelu':
        act_func = make_initializer(LazyPReLU, init_neg_slope=1)
    elif activation == 'tanhshrink':
        act_func = nn.Tanhshrink
    elif activation == 'tanh':
        act_func = nn.Tanh
    elif activation == 'softsign':
        act_func = nn.Softsign
    elif activation == 'softplus':
        act_func = make_initializer(nn.Softplus, beta=1)
    elif activation == 'symsum':
        act_func = SymSum
    elif activation == 'leakyrelu':
        act_func = make_initializer(nn.LeakyReLU, negative_slope=0.2)
    elif activation == 'abs':
        act_func = torch.abs
    elif activation == 'softshrink_hardtanh':
        act_func = make_initializer(SoftShrinkHardTanh, softshrink_coef=1)
    elif activation == 'softsymsum':
        act_func = make_initializer(SoftSymSum, beta=3)
    elif activation == 'swish':
        act_func = MemoryEfficientSwish
    else:
        raise ValueError(f'{activation} is not implemented.')

    return act_func
