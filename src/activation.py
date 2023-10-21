"""
Activation functions used in transformers. (as mentioned in https://arxiv.org/pdf/2002.05202v1.pdf)

This module contains the activation functions used in various transformer papers including:
1) GELU
2) Swish
3) GLU
4) Bilinear
5) ReGLU
6) GEGLU
7) SwiGLU
"""

from torch import erf, pow, split
from torch import sigmoid, tanh
from torch import Tensor
from torch.nn import Module, ReLU


M_SQRT1_2=0.70710678118654752440
M_SQRT2=1.41421356237309504880
M_2_SQRTPI=1.12837916709551257390


class GELU(Module):
    r""" Applies the Gaussian Error Linear Units function:

    Math:
        Exact:
            output = x * \Phi(x)
            Phi - Cumulative Distribution Function for Gaussian Distribution

        Approximation:
            output = 0.5 * x * (1 + Tanh(\sqrt{2 / pi} * (x + 0.044715 * x^3)))

    Args:
        approx(str, optional): GELU approximation algorithm to use.
                               ['none', 'tanh']
                               Default: 'none'

    Example:
        >>> gelu = GELU()
        >>> x = torch.randn(5)
        >>> output = gelu(x)
    """
    __constants__ = ['approx']
    approx: str
    def __init__(self, approx: str = 'none') -> None:
        super().__init__()
        self.approx = approx.lower()
        assert self.approx in ['none', 'tanh'], "Parameter approx takes values in ['none', 'tanh']"

    def forward(self, x: Tensor) -> Tensor:
        if self.approx == 'tanh':
            self.const = 0.5 * M_SQRT2 * M_2_SQRTPI  # sqrt(2/pi)
            return 0.5 * x * (1 + tanh(self.const * (x + 0.044715 * pow(x, 3))))
        # else (none)
        return 0.5 * x * ( 1 +  erf(0.70710678118654752440 * x))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(approx= \'{self.approx}\')"


class Swish(Module):
    r""" Applies the Sigmoid Linear Unit (SiLU)/swish function, element-wise.

    Math:
        output = x * \sigma(x)
        sigma - logistic sigmoid

    Example:
        >>> swish = Swish()
        >>> x = torch.randn(5)
        >>> output = swish(x)
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        return x * sigmoid(x)


class GLU(Module):
    r""" The gated linear unit

    Math:
        output = a * \sigma(b)
        where input is split in 2 halves a, b along dimension = dim
        sigma - logistic sigmoid

    Args:
        dim(int, optional): Dimension along which the input tensor is split into half
                            Default: -1

    Example:
        >>> glu = GLU()
        >>> x = torch.randn(5, 4)
        >>> output = glu(x)
    """
    def __init__(self, dim: int=-1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = split(x, 2, dim= self.dim)
        return a * sigmoid(b)
    

class Bilinear(Module):
    r"""" The gated linear unit without any activation

    Math:
        output = a * b
        where input is split in 2 halves a, b along dimension = dim

    Args:
        dim(int, optional): Dimension along which the input tensor is split into half
                            Default: -1

    Example:
        >>> bl = Bilinear()
        >>> x = torch.randn(5, 4)
        >>> output = bl(x)
    """
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = split(x, 2, dim= self.dim)
        return a * b


class ReGLU(Module):
    r"""" The gated linear unit with relu activation

    Math:
        output = a * \relu(b)
        where input is split in 2 halves a, b along dimension = dim
        where relu - rectified linear unit[max(0, b)]

    Args:
        dim(int, optional):      Dimension along which the input tensor is split into half
                                 Default: -1

        inplace(bool, optional): Input for relu, to be calculated in place
                                 Default: True

    Example:
        >>> reglu = ReGLU()
        >>> x = torch.randn(5, 4)
        >>> output = reglu(x)
    """
    def __init__(self, dim:int = -1, inplace:bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.relu = ReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        a, b = split(x, 2, dim= self.dim)
        return a * self.relu(b)


class GEGLU(Module):
    r"""" The gated linear unit with gelu activation

    Math:
        output = a * \gelu(b)
        where input is split in 2 halves a, b along dimension = dim
        where gelu - Gaussian Error Linear Units

    Args:
        dim(int, optional):    Dimension along which the input tensor is split into half
                               Default: -1

        approx(str, optional): GELU approximation algorithm to use.
                               ['none', 'tanh']
                               Default: 'none'

    Example:
        >>> geglu = GEGLU()
        >>> x = torch.randn(5, 4)
        >>> output = geglu(x)
    """
    def __init__(self, dim:int = -1, approx: str = 'none') -> None:
        super().__init__()
        self.dim = dim
        self.gelu = GELU(approx=approx)

    def forward(self, x: Tensor) -> Tensor:
        a, b = split(x, 2, dim= self.dim)
        return a * self.gelu(b)


class SwiGLU(Module):
    r"""" The gated linear unit with swish activation

    Math:
        output = a * \swish(b)
        where input is split in 2 halves a, b along dimension = dim
        where swish - Sigmoid Linear Unit (SiLU)/swish function

    Args:
        dim(int, optional): Dimension along which the input tensor is split into half
                            Default: -1

    Example:
        >>> swiglu = SwiGLU()
        >>> x = torch.randn(5, 4)
        >>> output = swiglu(x)
    """
    def __init__(self, dim:int = -1) -> None:
        super().__init__()
        self.dim = dim
        self.swish = Swish()

    def forward(self, x: Tensor) -> Tensor:
        a, b = split(x, 2, dim= self.dim)
        return a * self.swish(b)
