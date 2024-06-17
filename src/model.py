from torch import ones, rsqrt
from torch import Tensor, randn
from torch.nn import Linear, Module, Parameter
from torch.nn.functional import gelu, sigmoid

__all__ = ['activation', 'RMSNorm', 'MLP']


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
        add_unit_offset (bool): Add unit offset to norm
                                This is from google gemma paper
                                (#https://github.com/google/gemma_pytorch)

    Example:
        >>> x = torch.randn((5, 2))
        >>> rmsnorm = RMSNorm(dim=2)
        >>> norm = rmsnorm(x)
    """
    def __init__(self, dim: int, eps: float = 1e-6,
                 add_unit_offset: bool = True) -> None:
        r"""
        Initialize RMSNorm layer
        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ones(dim))
        self.add_unit_offset = add_unit_offset

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
        norm = self._norm(x_fp32).type_as(x)
        if self.add_unit_offset:
            return (1 + self.weight) * norm
        return self.weight * norm


class MLP(Module):
    r"""
    MLP Layer for transformer.
    TBD - use fuse linear layer as used in Gemma(https://github.com/google/gemma_pytorch)

    Args:
        hid_dim (int) : Hidden dimension size
        use_geglu (bool) : Use geglu activation.
                           Default: False
                           We use Silu by default.
        use_tanh (bool) : The tanh approximation to be used
                          Dafault: True
                          Note:- it's ignored for Silu.
        mult (int) : Multiplier for calculating intermediate dim
                     Default: 4

    Example:
        >>> x = torch.randn((5,2))
        >>> mlp = MLP(2)
        >>> out = mlp(x) 
    """
    def __init__(self, hid_dim: int, use_geglu: bool = False,
                 use_tanh: bool = True, mult: int = 4) -> None:
        r""""
        Initilize MLP Layer
        """
        super().__init__()
        self.c_fc = Linear(hid_dim, hid_dim * mult)
        self.c_proj = Linear(mult * hid_dim, hid_dim)
        self.use_geglu = use_geglu
        self.use_tanh = use_tanh

    def forward(self, x):
        r"""
        Forward pass for MLP layer
        """
        x = self.c_fc(x)
        x = activation(x, self.use_geglu, self.use_tanh)
        return self.c_proj(x)
