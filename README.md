# lite-mamba

A minimal, pure-PyTorch version of Mamba with a multi-dilated causal depthwise conv front-end. No CUDA/Triton build needed; works on CPU or GPU with standard PyTorch ops.

## Install
```bash
pip install torch einops
pip install lite-mamba
```

## Usage
```python
from lite_mamba import Mamba
import torch

x = torch.randn(2, 128, 512)  # (batch, seq, d_model)
m = Mamba(d_model=512, d_conv=3, conv_dilations=(1,2,4,8))
y = m(x)
print(y.shape)  # (2, 128, 512)
```

### Conv front-end variants
- `PTCNMamba`: parallel TCN branches (same as `Mamba`).
- `STCNMamba`: stacked TCN layers (sequential dilated convs).
- `DPWCMamba`: depthwise + pointwise conv branches.

## API quick reference
`Mamba(d_model, d_state=16, d_conv=4, conv_dilations=(1,), expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=False, layer_idx=None, device=None, dtype=None)`

- `d_model` (int, required): input/output embedding size.
- `d_state` (int, default 16): SSM state dimension per channel. Larger gives longer memory; increases compute.
- `d_conv` (int, default 4): depthwise conv kernel size for each branch.
- `conv_dilations` (tuple[int], default `(1,)`): dilation per branch. Multiple values create parallel dilated convs; effective receptive field is `(d_conv-1)*dilation`.
- `expand` (float, default 2): inner width multiplier; sets `d_inner = expand * d_model`.
- `dt_rank` (int or "auto", default "auto"): rank of delta projection. "auto" sets `ceil(d_model/16)`.
- `dt_min`, `dt_max` (float, defaults 1e-3 / 1e-1): log-uniform range for delta initialization.
- `dt_init` ("random" | "constant", default "random") and `dt_scale`, `dt_init_floor`: control delta init magnitude/stability.
- `conv_bias` (bool, default True): include bias in depthwise convs.
- `bias` (bool, default False): include bias in input/output linear projections.
- `use_fast_path` (bool): ignored in this pure-PyTorch build; kept for API compatibility.
- `layer_idx` (int | None): identifier for streaming cache registration; required when using `allocate_inference_cache` + `inference_params`.
- `device`, `dtype`: standard module factory kwargs.

### Inference / streaming helpers
- `allocate_inference_cache(batch_size, max_seqlen, dtype=None)`: preallocates conv and SSM state buffers for step-wise decoding.
- `step(hidden_states, conv_state, ssm_state)`: single-token forward (expects `hidden_states` with shape `(B, 1, d_model)`).
- `forward(..., inference_params)`: if `inference_params` has cached states (with `key_value_memory_dict` and `seqlen_offset`), uses them for streaming.

## Highlights
- Multi-branch causal dilated convs (weighted sum via learned gates).
- Pure Python: no custom C++/CUDA or Triton kernels.
- Streaming support via per-branch conv states and SSM state caching.

## Practical setups
- **Local modeling / small context**: `d_conv=3`, `conv_dilations=(1,2,4)`, `d_state=8–16`, `expand=2`.
- **Longer context**: widen `conv_dilations` (e.g., `(1,2,4,8,16)`) or increase `d_state` to 32; expect higher memory/compute.
- **Streaming/AR decoding**: call `allocate_inference_cache` once per layer, pass `inference_params` during forward; use `step` inside your generation loop.
- **Stability first**: keep `dt_min` >= 1e-4 and `dt_init_floor` small; leave defaults unless you observe drift or exploding activations.

## Notes
- Set different `conv_dilations` to adjust receptive field; keep kernels small (e.g., 3–5) to avoid excessive padding.
- `use_fast_path` flag is ignored here (kept for API compatibility).
- Reference selective scan is implemented in PyTorch for portability; faster fused kernels are omitted intentionally.

## License
Apache-2.0
