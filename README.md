# lite-mamba

[![Publish Python Package](https://github.com/Mrrobi/lite_mamba/actions/workflows/publish.yml/badge.svg)](https://github.com/Mrrobi/lite_mamba/actions/workflows/publish.yml)
[![Tests](https://github.com/Mrrobi/lite_mamba/actions/workflows/tests.yml/badge.svg)](https://github.com/Mrrobi/lite_mamba/actions/workflows/tests.yml)

A minimal, pure-PyTorch implementation of Mamba with a multi-dilated causal depthwise conv front-end. No CUDA/Triton build needed; works on CPU or GPU with standard PyTorch ops.

**Dual Framework Support**: Includes both PyTorch and TensorFlow implementations with identical core logic and mathematical formulations, so you can use either framework depending on your environment.

## Install
```bash
pip install torch einops
pip install lite-mamba
```

TensorFlow path (optional):
```bash
pip install "lite-mamba[tensorflow]"
```

## Usage

### PyTorch
```python
from lite_mamba import Mamba, PTCNMamba, STCNMamba, DPWCMamba
import torch

x = torch.randn(2, 128, 512)  # (batch, seq, d_model)
m = Mamba(d_model=512, d_conv=3, conv_dilations=(1, 2, 4, 8))
y = m(x)
print(y.shape)  # (2, 128, 512)
```

### TensorFlow
```python
from lite_mamba import TFPTCNMamba
import tensorflow as tf

x = tf.random.normal((2, 128, 512))  # (batch, seq, d_model)
m = TFPTCNMamba(d_model=512, d_conv=3, conv_dilations=(1, 2, 4, 8))
y = m(x)
print(y.shape)  # (2, 128, 512)
```

### Conv front-end variants
- `PTCNMamba`: identical to `Mamba`, mixes parallel dilated depthwise conv branches via learned softmax gates.
- `STCNMamba`: runs the same depthwise conv layers in sequence (no gating); each branch output feeds the next to create a deterministic dilation stack.
- `DPWCMamba`: pairs each depthwise branch with a pointwise (1×1) conv before the gating mix, adding extra channel mixing without stacking more layers.

All variants expose the same constructor signature (`d_model`, `d_state`, `conv_dilations`, etc.) and streaming helpers (`allocate_inference_cache`, `step`). Swap them simply by changing the imported class name:

```python
from lite_mamba import STCNMamba

m = STCNMamba(d_model=512, d_state=16, conv_dilations=(1, 2, 4))
```

Use `DPWCMamba` for richer channel interactions in each branch, and `STCNMamba` when you want a straightforward sequential dilation pipeline (e.g., for debugging or reproducing the behavior of stacked TCN layers).

### Baseline helper
`BaselineMamba` mirrors the upstream `state-spaces/mamba` block: a single depthwise causal convolution followed by the SSM parameter projection, selective scan recurrence, and streaming helpers. `baseline_mamba` is a thin functional alias that instantiates the class with the same defaults so you can reproduce the reference layout without duplicating constructor arguments.

```python
from lite_mamba import BaselineMamba

m = BaselineMamba(d_model=512, d_conv=3)
```

### TensorFlow variants
TensorFlow implementations mirror the exact same core logic and mathematical formulations as PyTorch:
- `TFBaselineMamba` - Single-branch baseline matching reference Mamba
- `TFPTCNMamba` - Parallel dilated TCN branches with learned gating (default `TFMamba`)
- `TFSTCNMamba` - Stacked/sequential dilated TCN branches
- `TFDPWCMamba` - Depthwise + pointwise convolution branches

All TensorFlow variants support:
- Same constructor API as PyTorch versions
- Streaming inference via `allocate_inference_cache()` and `step()`
- Custom dilated causal convolutions
- Identical SSM discretization and selective scan

```python
import tensorflow as tf
from lite_mamba import TFBaselineMamba, TFSTCNMamba, TFDPWCMamba

# Baseline single-branch
m1 = TFBaselineMamba(d_model=512, d_conv=3)

# Stacked dilated branches
m2 = TFSTCNMamba(d_model=512, d_conv=3, conv_dilations=(1, 2, 4))

# Depthwise + pointwise
m3 = TFDPWCMamba(d_model=512, d_conv=3, conv_dilations=(1, 2, 4, 8))
```

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
- **Dual Framework**: PyTorch and TensorFlow implementations with identical core logic
- **Multi-branch Architecture**: Parallel or stacked causal dilated convolutions with learned gating
- **Pure Python**: No custom C++/CUDA or Triton kernels required
- **Streaming Support**: Per-branch conv states and SSM state caching for autoregressive generation
- **Framework Parity**: Mathematically equivalent implementations verified across both frameworks

## Practical setups
- **Local modeling / small context**: `d_conv=3`, `conv_dilations=(1,2,4)`, `d_state=8–16`, `expand=2`.
- **Longer context**: widen `conv_dilations` (e.g., `(1,2,4,8,16)`) or increase `d_state` to 32; expect higher memory/compute.
- **Streaming/AR decoding**: call `allocate_inference_cache` once per layer, pass `inference_params` during forward; use `step` inside your generation loop.
- **Stability first**: keep `dt_min` >= 1e-4 and `dt_init_floor` small; leave defaults unless you observe drift or exploding activations.

## Framework Compatibility

### PyTorch vs TensorFlow
Both implementations are mathematically equivalent and follow the same core Mamba architecture:
- **Tensor Layout**: PyTorch uses `(B, D, L)` (channels-first), TensorFlow uses `(B, L, D)` (channels-last)
- **SSM Formulation**: Identical selective state space model with zero-order hold discretization
- **Initialization**: Same parameter initialization schemes (A matrix, D parameter, dt bias)
- **Variants**: All architectural variants available in both frameworks

The implementations have been thoroughly tested to ensure numerical equivalence within floating-point precision.

## Notes
- Set different `conv_dilations` to adjust receptive field; keep kernels small (e.g., 3–5) to avoid excessive padding.
- `use_fast_path` flag is ignored here (kept for API compatibility with reference implementations).
- Reference selective scan is implemented in pure Python for portability; faster fused kernels are omitted intentionally.
- TensorFlow implementation uses custom depthwise convolution via `tf.nn.depthwise_conv2d` for dilated causal convolutions.

## License
Apache-2.0
