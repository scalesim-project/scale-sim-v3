"""Slow Mitchell layers for checking ResNet-18 accuracy."""
import torch
from torch import nn
from torch.nn import functional as F


def mitchell_torch(a, b):
    """Mitchell multiply with broadcasting, including negative values and zero."""
    sign = torch.sign(a) * torch.sign(b)
    x = a.abs()
    y = b.abs()
    # Clamp only for the log, zero inputs still produce zero below
    safe_x = x.clamp_min(1e-30)
    safe_y = y.clamp_min(1e-30)
    ka = torch.floor(torch.log2(safe_x))
    kb = torch.floor(torch.log2(safe_y))
    xa = safe_x / torch.exp2(ka) - 1
    xb = safe_y / torch.exp2(kb) - 1
    f = xa + xb
    z = torch.where(f < 1, torch.exp2(ka + kb) * (1 + f), torch.exp2(ka + kb + 1) * f)
    return torch.where((x == 0) | (y == 0), torch.zeros_like(z), sign * z)


def mitchell_matrix(a, w, out_chunk=8, pos_chunk=16, k_chunk=64):
    """Multiply (K,P) with (O,K) using small chunks to avoid huge tensors."""
    k, positions = a.shape
    outputs = w.shape[0]
    result = a.new_zeros((outputs, positions))
    for o in range(0, outputs, out_chunk):
        for p in range(0, positions, pos_chunk):
            part = result[o:o + out_chunk, p:p + pos_chunk]
            for kk in range(0, k, k_chunk):
                aa = a[kk:kk + k_chunk, p:p + pos_chunk]
                ww = w[o:o + out_chunk, kk:kk + k_chunk]
                part += mitchell_torch(ww[:, :, None], aa[None, :, :]).sum(dim=1)
    return result


class MitchellLinear(nn.Module):
    """Linear layer using approximate products, not changed weights."""

    def __init__(self, layer):
        super().__init__()
        self.register_buffer("weight", layer.weight.detach().clone())
        self.register_buffer("bias", layer.bias.detach().clone() if layer.bias is not None else None)

    def forward(self, x):
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).T
        out = mitchell_matrix(flat, self.weight).T
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape, self.weight.shape[0])


class MitchellConv2d(nn.Module):
    """Convolution using unfolded patches and approximate multiplication."""

    def __init__(self, layer):
        super().__init__()
        self.register_buffer("weight", layer.weight.detach().clone())
        self.register_buffer("bias", layer.bias.detach().clone() if layer.bias is not None else None)
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups
        self.padding_mode = layer.padding_mode
        self.out_channels = layer.out_channels

    def forward(self, x):
        if self.padding_mode != "zeros":
            raise ValueError("MitchellConv2d currently supports zero padding only")
        h, w = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        oh = (h + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        ow = (w + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        result = []
        for sample in x:
            patches = F.unfold(sample[None], (kh, kw), dilation=self.dilation,
                               padding=self.padding, stride=self.stride)[0]
            per_group = patches.shape[0] // self.groups
            out_per_group = self.out_channels // self.groups
            parts = []
            for g in range(self.groups):
                aa = patches[g * per_group:(g + 1) * per_group]
                ww = self.weight[g * out_per_group:(g + 1) * out_per_group].reshape(out_per_group, -1)
                parts.append(mitchell_matrix(aa, ww))
            out = torch.cat(parts, dim=0)
            if self.bias is not None:
                out += self.bias[:, None]
            result.append(out.reshape(self.out_channels, oh, ow))
        return torch.stack(result)


def replace_with_mitchell(model):
    """Replace all Conv2d and Linear layers in a model."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(model, name, MitchellConv2d(child))
        elif isinstance(child, nn.Linear):
            setattr(model, name, MitchellLinear(child))
        else:
            replace_with_mitchell(child)
    return model
