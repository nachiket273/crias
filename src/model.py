from torch import ones, rsqrt
from torch import Tensor, randn
from torch.nn import Module, Parameter
from torch.nn.functional import gelu, sigmoid

__all__ = ['activation', 'RMSNorm']


def activation(x: Tensor,
               use_geglu: bool = False,
               use_tanh: bool = False) -> Tensor:
    r"""
    Computes the activation function.
    The literature uses different activations from gelu, to swiglu.
    We'll here use either geglu from phi-3 (https://arxiv.org/pdf/2404.14219)
    or swiglu/silu from llama, openelm, others.

    Args:
        x (Tensor): Input tensor
        use_geglu (bool): Use geglu activation.
                          If false, swiglu/silu will be used.
                          Deafult: False
        use_tanh (bool): The tanh approximation to be used.
                         If false, 'none' approximation will be used.
                         Default: False

    Returns:
        Tensor post activation calculation.

    Examples:
        >>> x = Tensor((5, 2))

        1. Use SwiGLU:
            >>> out = activation(x)

        2. Use GeGLU with 'none' approximation:
            >>> out = activation(x, use_geglu=True)

        3. Use GeGLU with 'tahn' approximation:
            >>> out = activation(x, use_geglu=True, use_tanh=True)
    """
    if use_geglu:
        if use_tanh:
            return x * gelu(x, approximate='tanh')
        else:
            return x * gelu(x, approximate='none')
    else:
        return x * sigmoid(x)


class RMSNorm(Module):
    r""""
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/pdf/1910.07467.pdf.

    Implementation based on:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int) : Embedding dimension
        eps (float) : Epsilon avoid divide by zero
                      Default: 1e-6

    Example:
        >>> x = torch.randn((5, 2))
        >>> rmsnorm = RMSNorm(dim=2)
        >>> norm = rmsnorm(x)
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        r"""
        Initialize RMSNorm layer
        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        r""""
        Calculate the norm for the input tensor.
        """
        return x * rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        r""""
        Forward pass through RMSNorm.
        """
        x_fp32 = x.float()
        return self.weight * self._norm(x_fp32).type_as(x)
