import torch
from torch import nn
from torch.nn import functional as F

class Similarity(nn.Module):
    """Compute the cosine similarity between two tensors along the last dimension.
    If normalize is True, the input tensors are normalized to have L2 norm of 1.
    Args:
        a (torch.Tensor): Shape of (batch_size,seq_len, d_model)
        b (torch.Tensor): Shape of (batch_size,seq_len, d_model)
    Returns:
        torch.Tensor: Shape of (batch_size, seq_len, seq_len)
    """
    def __init__(self, eps = 1e-8, normalize = True):
        super(Similarity, self).__init__()
        self.eps = eps
        self.normalize = normalize

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            a = F.normalize(a, p=2, dim=-1, eps = self.eps)
            b = F.normalize(b, p=2, dim=-1, eps = self.eps)
        return (a @ b.transpose(-1,-2))
    

class SelfSimilarity(nn.Module):
    """Compute the cosine similarity between a tensor and itself along the last dimension.
    If normalize is True, the input tensor is normalized to have L2 norm of 1.
    Args:
        x (torch.Tensor): Shape of (batch_size,seq_len, d_model)
    Returns:
        torch.Tensor: Shape of (batch_size, seq_len, seq_len)
        """
    def __init__(self, eps = 1e-8, normalize = True):
        super(SelfSimilarity, self).__init__()
        self.eps = eps
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1, eps = self.eps)
        return (x @ x.transpose(-1,-2))