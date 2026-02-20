# lite-mamba

[![Publish Python Package](https://github.com/Mrrobi/lite_mamba/actions/workflows/publish.yml/badge.svg)](https://github.com/Mrrobi/lite_mamba/actions/workflows/publish.yml)
[![Tests](https://github.com/Mrrobi/lite_mamba/actions/workflows/tests.yml/badge.svg)](https://github.com/Mrrobi/lite_mamba/actions/workflows/tests.yml)

A minimal, pure-TensorFlow implementation of Mamba with a multi-dilated causal depthwise conv front-end. No custom C++ or Triton kernels needed; works seamlessly on CPU, GPU, or TPU with standard TensorFlow ops.

## Install
```bash
pip install lite-mamba
```

## Usage

```python
from lite_mamba import TFPTCNMamba
import tensorflow as tf

x = tf.random.normal((2, 128, 512))  # (batch, seq, d_model)
m = TFPTCNMamba(d_model=512, d_conv=3, conv_dilations=(1, 2, 4, 8))
y = m(x)
print(y.shape)  # (2, 128, 512)
```

### Conv front-end variants
- `TFPTCNMamba`: default Mamba variant, mixes parallel dilated depthwise conv branches via learned softmax gates.
- `TFSTCNMamba`: runs the same depthwise conv layers in sequence (no gating); each branch output feeds the next to create a deterministic dilation stack.
- `TFDPWCMamba`: pairs each depthwise branch with a pointwise (1x1) conv before the gating mix, adding extra channel mixing without stacking more layers.
- `TFBaselineMamba`: single-branch baseline matching the reference Mamba architecture from state-spaces.

All variants expose the same constructor signature (`d_model`, `d_state`, `conv_dilations`, etc.) and streaming helpers (`allocate_inference_cache`, `step`). Swap them simply by changing the imported class name:

```python
from lite_mamba import TFSTCNMamba

m = TFSTCNMamba(d_model=512, d_state=16, conv_dilations=(1, 2, 4))
```

## API quick reference
`TFMamba(d_model, d_state=16, d_conv=4, conv_dilations=(1,), expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, layer_idx=None)`

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
- `layer_idx` (int | None): identifier for streaming cache registration.

### Inference / streaming helpers
- `allocate_inference_cache(batch_size, max_seqlen, dtype=None)`: preallocates conv and SSM state buffers for step-wise decoding.
- `step(hidden_states, conv_state, ssm_state)`: single-token forward (expects `hidden_states` with shape `(B, 1, d_model)`).

## Highlights
- **Multi-branch Architecture**: Parallel or stacked causal dilated convolutions with learned gating.
- **Pure TensorFlow**: No custom C++/CUDA kernels required. Compatible with XLA compilation (`tf.function(jit_compile=True)`).
- **Streaming Support**: Per-branch conv states and SSM state caching for autoregressive generation.

## License
Apache-2.0
