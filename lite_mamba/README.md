# lite-mamba

A minimal, pure-PyTorch version of Mamba with a multi-dilated causal depthwise conv front-end. No CUDA/Triton build needed; works on CPU or GPU with standard PyTorch ops.

## Install
```bash
pip install torch einops
pip install .  # from repo root containing pyproject.toml
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

## Highlights
- Multi-branch causal dilated convs (weighted sum via learned gates).
- Pure Python: no custom C++/CUDA or Triton kernels.
- Streaming support via per-branch conv states and SSM state caching.

## Notes
- Set different `conv_dilations` to adjust receptive field; keep kernels small (e.g., 3–5).
- `use_fast_path` flag is ignored here (kept for API compatibility).
- Reference selective scan is implemented in PyTorch for portability; faster fused kernels are omitted intentionally.

## License
Apache-2.0
