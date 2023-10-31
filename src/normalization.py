"""
Implements normalization techniques used in transformers, especially
1) Layer Normalization
2) RMSNorm (https://arxiv.org/pdf/1910.07467.pdf)
"""
from torch import ones, zeros
from torch import norm, split, sqrt
from torch import Tensor
from torch.nn import Module, Parameter


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
                                  (RMSNorm doesn't enforce re-centering invariance)
    """
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
