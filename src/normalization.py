"""
Implements normalization techniques used in transformers, especially
1) Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)
2) RMSNorm (https://arxiv.org/pdf/1910.07467.pdf)
"""
from torch import ones, zeros
from torch import norm, split
from torch import Size, Tensor
from torch.nn import Module, Parameter
from typing import List, Union

__all__ = ['RMSNorm', 'LayerNorm']

input_shape = Union[int, List[int], Size]

class RMSNorm(Module):
    r""" Applies RMSNorm to the input of the layer

    Args:
        dim(int, required) :      The model dimesion
        partial(float, optional): The partial RMSNorm value. [0, 1]
                                  Default: -1 (full RMSNorm/partial disabled)
        eps(float, optional):     The epsilon value for arithmetic stability.
                                  Default: 1e-8
        bias(bool, optional):     Whether to use bias for RMSNorm.
                                  Default: False
                                  
    Example:
        >>> rmsnorm = RMSNorm(4)
        >>> x = torch.randn((5, 4))
        >>> output = rmsnorm(x)
    """
    __constants__ = ['dim', 'partial', 'eps', 'bias']
    dim: int
    partial: int
    eps: float
    bias: bool

    def __init__(self, dim: int, partial: int=-1, eps: float = 1e-8, bias: bool=False) -> None:
        super().__init__()
        self.dim = dim
        self.partial = partial
        if self.partial != -1 and (self.partial < 0 or self.partial > 1):
            raise ValueError("Value of parameter partial should be in range [0,1]")
        self.eps = eps
        self.bias = bias
        self.__init_parameters__()

    def __init_parameters__(self) -> None:
        self.scale = Parameter(ones(self.dim))
        self.register_parameter('scale', self.scale)
        if self.bias:
            self.off = Parameter(zeros(self.dim))
            self.register_parameter('offset', self.off)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, partial={self.partial}, eps={self.eps}, bias={self.bias})"

    def forward(self, x: Tensor) -> Tensor:
        if self.partial != -1:
            # Partial RMSNorm
            dimx = int(self.dim * self.partial)
            if dimx == 0:
                dimx = 1
            partial_x, _ = split(x, [dimx, self.dim - dimx], dim=-1)
            normx = norm(partial_x, 2, dim=-1, keepdim=True)
        else:
            normx = norm(x, 2, dim=-1, keepdim=True)
            dimx = self.dim

        rms = normx  *  dimx ** (-1./2)
        x_normed = x / (rms + self.eps)

        if self.bias:
            return self.scale * x_normed + self.off
        
        return self.scale * x_normed


class LayerNorm(Module):
    r""" Applies Layer Normalization to the input of the layer.

    Args:
        shape(int or list or torch.Size, required): Input shape from an expected input of size
        eps(float, optional):                       The epsilon value for arithmetic stability.
                                                    Default: 1e-8
        gamma(bool, optional):                      Add scale parameter.
                                                    Default: True
        offset(bool, optional):                     Add bias parameter.
                                                    Default: True
    
    Example:
        >>> lnorm = LayerNorm(4)
        >>> x = torch.randn((5, 4))
        >>> output = lnorm(x)
    """
    __constants__ = ['shape', 'eps', 'gamma', 'offset']
    shape: input_shape
    eps: float
    gamma: bool
    offset: bool

    def __init__(self, shape: input_shape, eps: float=1e-8, gamma: bool=True, offset: bool=True) -> None:
        super().__init__()
        if isinstance(shape, int):
            self.shape = (shape, )
        else:
            self.shape = (shape[-1], )
        self.shape = Size(self.shape)
        self.eps = eps
        self.gamma = gamma
        self.offset = offset
        self.__init_parameters__()

    def __init_parameters__(self) -> None:
        if self.gamma:
            self.scale = Parameter(ones(self.shape))
            self.register_parameter('scale', self.scale)
        if self.offset:
            self.bias = Parameter(zeros(self.shape))
            self.register_parameter('bias', self.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, eps={self.eps}, gamma={self.gamma}, offset={self.offset})"
    
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        # var = x.var(dim=-1, keepdim=True)
        var = ((x - mean)**2).mean(dim=-1, keepdim=True)
        y = (x - mean) / (var + self.eps).sqrt()
        if self.gamma:
            y *= self.scale
        if self.offset:
            y += self.bias
        return y
