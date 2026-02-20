# lite-mamba

[![Publish Python Package](https://github.com/Mrrobi/lite_mamba/actions/workflows/publish.yml/badge.svg)](https://github.com/Mrrobi/lite_mamba/actions/workflows/publish.yml)
[![Tests](https://github.com/Mrrobi/lite_mamba/actions/workflows/tests.yml/badge.svg)](https://github.com/Mrrobi/lite_mamba/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-gpuon.me-blue.svg)](https://gpuon.me/lite_mamba/)

A minimal, pure-TensorFlow implementation of Mamba with a multi-dilated causal depthwise conv front-end. No custom C++ or Triton kernels needed; works seamlessly on CPU, GPU, or TPU with standard TensorFlow ops.

<br>

**📚 [READ THE FULL DOCUMENTATION HERE](https://gpuon.me/lite_mamba/) 📚**

Contains architecture details, API references, streaming inference guides, and overviews of the multi-branch variants (`TFPTCNMamba`, `TFSTCNMamba`, `TFDPWCMamba`).

---

## Quick Start

### Install
```bash
pip install lite-mamba
```

### Basic Usage

```python
from lite_mamba import TFPTCNMamba
import tensorflow as tf

x = tf.random.normal((2, 128, 512))  # (batch, seq, d_model)
m = TFPTCNMamba(d_model=512, d_conv=3, conv_dilations=(1, 2, 4, 8))

y = m(x)
print(y.shape)  # (2, 128, 512)
```

## License
Apache-2.0
