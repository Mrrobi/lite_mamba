import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from .selective_scan import selective_scan_fn
from .causal_dilated_conv1d import causal_dilated_conv1d_fn, causal_dilated_conv1d_update


class Mamba(nn.Module):
    # Base Mamba block with parallel dilated depthwise conv branches.
    # Shapes use: B=batch, L=sequence length, D=d_model, Di=d_inner.
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        conv_dilations=(1,),
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_dilations = tuple(conv_dilations)
        self.conv_bias = conv_bias
        self.num_conv_branches = len(self.conv_dilations)
        self.conv_state_lens = [(self.d_conv - 1) * d + 1 for d in self.conv_dilations]
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        # Project input to conv stream (x) and gating stream (z).
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Parallel depthwise causal conv branches (one per dilation).
        self.conv1d_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    groups=self.d_inner,
                    padding=d * (d_conv - 1),
                    dilation=d,
                    **factory_kwargs,
                )
                for d in self.conv_dilations
            ]
        )
        self.conv1d = self.conv1d_layers[0]
        # Learned mixing weights over conv branches.
        self.conv_gates = nn.Parameter(torch.ones(self.num_conv_branches))

        self.activation = "silu"
        self.act = nn.SiLU()

        # Project conv output to SSM parameters (dt, B, C).
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # Delta projection for discretization.
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # Project SSM output back to model dim.
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        # hidden_states: (B, L, D)
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            # Streaming mode: reuse cached conv/SSM states.
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # (B, L, D) -> (B, Di*2, L)
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        use_multi_branch = self.num_conv_branches > 1

        # Split into conv stream x and gating stream z.
        x, z = xz.chunk(2, dim=1)
        # Run parallel causal conv branches and mix them.
        conv_outputs = []
        if conv_state is not None:
            for state, conv_layer, dilation in zip(conv_state, self.conv1d_layers, self.conv_dilations):
                # Update cached conv state for streaming.
                pad_len = state.shape[-1] - x.shape[-1]
                if pad_len >= 0:
                    state.copy_(F.pad(x, (pad_len, 0)))
                else:
                    state.copy_(x[..., -state.shape[-1] :])
                if dilation > 1:
                    # Custom causal dilated depthwise conv for d>1.
                    xi = causal_dilated_conv1d_fn(
                        x=x,
                        weight=rearrange(conv_layer.weight, "d 1 w -> d w"),
                        bias=conv_layer.bias,
                        activation=self.activation,
                        dilation=dilation,
                    )
                else:
                    # Use built-in conv for dilation=1.
                    xi = self.act(conv_layer(x)[..., :seqlen])
                conv_outputs.append(xi)
        else:
            for conv_layer, dilation in zip(self.conv1d_layers, self.conv_dilations):
                if dilation > 1:
                    xi = causal_dilated_conv1d_fn(
                        x=x,
                        weight=rearrange(conv_layer.weight, "d 1 w -> d w"),
                        bias=conv_layer.bias,
                        activation=self.activation,
                        dilation=dilation,
                    )
                else:
                    xi = self.act(conv_layer(x)[..., :seqlen])
                conv_outputs.append(xi)
        # Softmax gates across branches and combine.
        gate = torch.softmax(self.conv_gates, dim=0)
        x = sum(g * xi for g, xi in zip(gate, conv_outputs))

        # Project to SSM parameters per position.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # Run selective scan (SSM recurrence over sequence).
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            # Update cached SSM state for streaming.
            y, last_state = y
            ssm_state.copy_(last_state)
        # Back to (B, L, Di) then project to (B, L, D).
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # Single-token step; hidden_states: (B, 1, D).
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        conv_outputs = []
        for state, conv_layer, dilation in zip(conv_state, self.conv1d_layers, self.conv_dilations):
            if dilation > 1:
                # Causal dilated update using rolling state buffer.
                xi = causal_dilated_conv1d_update(
                    x,
                    state,
                    rearrange(conv_layer.weight, "d 1 w -> d w"),
                    conv_layer.bias,
                    self.activation,
                    dilation=dilation,
                )
            else:
                # Manual depthwise conv update for dilation=1.
                state.copy_(torch.roll(state, shifts=-1, dims=-1))
                state[:, :, -1] = x
                xi = torch.sum(state * rearrange(conv_layer.weight, "d 1 w -> d w"), dim=-1)
                if conv_layer.bias is not None:
                    xi = xi + conv_layer.bias
                xi = self.act(xi).to(dtype=dtype)
            conv_outputs.append(xi)
        # Mix branches with softmax gates.
        gate = torch.softmax(self.conv_gates, dim=0)
        x = sum(g * xi for g, xi in zip(gate, conv_outputs))

        # SSM update for one step.
        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = [
            torch.zeros(
                batch_size,
                self.d_model * self.expand,
                state_len,
                device=device,
                dtype=conv_dtype,
            )
            for state_len in self.conv_state_lens
        ]
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state


class PTCNMamba(Mamba):
    """Parallel TCN branches (current Mamba behavior)."""


class STCNMamba(Mamba):
    """Stacked TCN layers (sequential causal convolutions)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Stacked variant doesn't use branch gating.
        self.conv_gates = None

    def forward(self, hidden_states, inference_params=None):
        # Same as base, but conv branches are applied sequentially.
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=1)
        # Stacked convs: output of each layer feeds the next.
        if conv_state is not None:
            for state, conv_layer, dilation in zip(
                conv_state, self.conv1d_layers, self.conv_dilations
            ):
                pad_len = state.shape[-1] - x.shape[-1]
                if pad_len >= 0:
                    state.copy_(F.pad(x, (pad_len, 0)))
                else:
                    state.copy_(x[..., -state.shape[-1] :])
                if dilation > 1:
                    x = causal_dilated_conv1d_fn(
                        x=x,
                        weight=rearrange(conv_layer.weight, "d 1 w -> d w"),
                        bias=conv_layer.bias,
                        activation=self.activation,
                        dilation=dilation,
                    )
                else:
                    x = self.act(conv_layer(x)[..., :seqlen])
        else:
            for conv_layer, dilation in zip(self.conv1d_layers, self.conv_dilations):
                if dilation > 1:
                    x = causal_dilated_conv1d_fn(
                        x=x,
                        weight=rearrange(conv_layer.weight, "d 1 w -> d w"),
                        bias=conv_layer.bias,
                        activation=self.activation,
                        dilation=dilation,
                    )
                else:
                    x = self.act(conv_layer(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # Single-token step with stacked convs.
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        for state, conv_layer, dilation in zip(conv_state, self.conv1d_layers, self.conv_dilations):
            if dilation > 1:
                x = causal_dilated_conv1d_update(
                    x,
                    state,
                    rearrange(conv_layer.weight, "d 1 w -> d w"),
                    conv_layer.bias,
                    self.activation,
                    dilation=dilation,
                )
            else:
                state.copy_(torch.roll(state, shifts=-1, dims=-1))
                state[:, :, -1] = x
                x = torch.sum(state * rearrange(conv_layer.weight, "d 1 w -> d w"), dim=-1)
                if conv_layer.bias is not None:
                    x = x + conv_layer.bias
                x = self.act(x).to(dtype=dtype)

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


class DPWCMamba(Mamba):
    """Depthwise + pointwise conv branches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        factory_kwargs = {
            "device": self.in_proj.weight.device,
            "dtype": self.in_proj.weight.dtype,
        }
        # Replace depthwise-only branches with depthwise + pointwise.
        del self.conv1d_layers
        del self.conv1d
        self.dw_conv1d_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    bias=self.conv_bias,
                    kernel_size=self.d_conv,
                    groups=self.d_inner,
                    padding=d * (self.d_conv - 1),
                    dilation=d,
                    **factory_kwargs,
                )
                for d in self.conv_dilations
            ]
        )
        self.pw_conv1d_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    bias=True,
                    kernel_size=1,
                    **factory_kwargs,
                )
                for _ in self.conv_dilations
            ]
        )
        self.conv1d = self.dw_conv1d_layers[0]

    def _pointwise_step(self, x, pw_conv):
        # Apply 1x1 pointwise conv to a (B, Di) tensor.
        weight = pw_conv.weight.squeeze(-1)
        bias = pw_conv.bias
        return F.linear(x, weight, bias)

    def forward(self, hidden_states, inference_params=None):
        # Same as base, but each branch is depthwise then pointwise.
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=1)
        conv_outputs = []
        if conv_state is not None:
            for state, dw_layer, pw_layer, dilation in zip(
                conv_state, self.dw_conv1d_layers, self.pw_conv1d_layers, self.conv_dilations
            ):
                pad_len = state.shape[-1] - x.shape[-1]
                if pad_len >= 0:
                    state.copy_(F.pad(x, (pad_len, 0)))
                else:
                    state.copy_(x[..., -state.shape[-1] :])
                if dilation > 1:
                    xi = causal_dilated_conv1d_fn(
                        x=x,
                        weight=rearrange(dw_layer.weight, "d 1 w -> d w"),
                        bias=dw_layer.bias,
                        activation=self.activation,
                        dilation=dilation,
                    )
                else:
                    xi = self.act(dw_layer(x)[..., :seqlen])
                # Pointwise 1x1 conv to mix channels.
                xi = pw_layer(xi)
                conv_outputs.append(xi)
        else:
            for dw_layer, pw_layer, dilation in zip(
                self.dw_conv1d_layers, self.pw_conv1d_layers, self.conv_dilations
            ):
                if dilation > 1:
                    xi = causal_dilated_conv1d_fn(
                        x=x,
                        weight=rearrange(dw_layer.weight, "d 1 w -> d w"),
                        bias=dw_layer.bias,
                        activation=self.activation,
                        dilation=dilation,
                    )
                else:
                    xi = self.act(dw_layer(x)[..., :seqlen])
                # Pointwise 1x1 conv to mix channels.
                xi = pw_layer(xi)
                conv_outputs.append(xi)
        gate = torch.softmax(self.conv_gates, dim=0)
        x = sum(g * xi for g, xi in zip(gate, conv_outputs))

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        conv_outputs = []
        for state, dw_layer, pw_layer, dilation in zip(
            conv_state, self.dw_conv1d_layers, self.pw_conv1d_layers, self.conv_dilations
        ):
            if dilation > 1:
                xi = causal_dilated_conv1d_update(
                    x,
                    state,
                    rearrange(dw_layer.weight, "d 1 w -> d w"),
                    dw_layer.bias,
                    self.activation,
                    dilation=dilation,
                )
            else:
                state.copy_(torch.roll(state, shifts=-1, dims=-1))
                state[:, :, -1] = x
                xi = torch.sum(state * rearrange(dw_layer.weight, "d 1 w -> d w"), dim=-1)
                if dw_layer.bias is not None:
                    xi = xi + dw_layer.bias
                xi = self.act(xi).to(dtype=dtype)
            xi = self._pointwise_step(xi, pw_layer).to(dtype=dtype)
            conv_outputs.append(xi)
        gate = torch.softmax(self.conv_gates, dim=0)
        x = sum(g * xi for g, xi in zip(gate, conv_outputs))

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = [
                torch.zeros(
                    batch_size,
                    self.d_model * self.expand,
                    state_len,
                    device=self.conv1d.weight.device,
                    dtype=self.conv1d.weight.dtype,
                )
                for state_len in self.conv_state_lens
            ]
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                for s in conv_state:
                    s.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
