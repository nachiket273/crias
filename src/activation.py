"""
Activation functions used in transformers. ( as mentioned in https://arxiv.org/pdf/2002.05202v1.pdf)

This module contains the activation functions used in various transformer papers including:
1) GELU
2) Swish
3) GLU
4) 
"""

from torch import erf, nn, pow, tanh, Tensor


M_SQRT1_2=0.70710678118654752440
M_SQRT2=1.41421356237309504880
M_2_SQRTPI=1.12837916709551257390

class GELU(nn.Module):
    r""" Applies the Gaussian Error Linear Units function:

    Args:
        approx(str, optional): gelu approximation algorithm to use.
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
    



  