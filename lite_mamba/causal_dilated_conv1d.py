import torch
import torch.nn.functional as F


def _apply_activation(x, activation):
    if activation is None or activation == "identity":
        return x
    if activation in ("silu", "swish"):
        return F.silu(x)
    if activation == "relu":
        return F.relu(x)
    raise ValueError(f"Unsupported activation: {activation}")


def causal_dilated_conv1d_fn(x, weight, bias=None, activation=None, dilation=1):
    """
    Depthwise causal 1D convolution with dilation.
    x: (B, D, L)
    weight: (D, W)
    bias: (D,) or None
    """
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    pad = dilation * (weight.shape[-1] - 1)
    x = F.pad(x, (pad, 0))
    y = F.conv1d(
        x,
        weight.unsqueeze(1),
        bias=bias,
        stride=1,
        padding=0,
        dilation=dilation,
        groups=weight.shape[0],
    )
    return _apply_activation(y, activation)


def causal_dilated_conv1d_update(x, conv_state, weight, bias=None, activation=None, dilation=1):
    """
    Single-step causal dilated conv update.
    x: (B, D)
    conv_state: (B, D, S) where S = dilation * (W - 1) + 1
    weight: (D, W)
    bias: (D,) or None
    """
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = x

    idx = torch.arange(0, weight.shape[-1], device=conv_state.device) * dilation
    pos = conv_state.shape[-1] - 1 - idx
    values = conv_state.index_select(-1, pos)
    y = torch.sum(values * weight.unsqueeze(0), dim=-1)
    if bias is not None:
        y = y + bias
    return _apply_activation(y, activation)
