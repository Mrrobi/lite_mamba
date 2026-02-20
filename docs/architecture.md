# Architecture & Variants

The `lite-mamba` library offers a pure-TensorFlow implementation of the Mamba architecture, featuring a unique multi-dilated causal depthwise convolution front-end. 

This diagram illustrates the core processing flow and how the different convolutional variants operate.

## Processing Flow

At a high level, the Mamba block performs the following sequence of operations:

1. **Input Projection**: The input sequence is linearly projected to expand the feature dimension.
2. **Convolutional Front-end**: The projected sequence passes through a causal 1D convolution layer. Here, `lite-mamba` introduces several multi-branch variants (detailed below) to replace the standard single depthwise convolution.
3. **SSM Parameter Projections**: The convolutional output is projected to form the parameters ($\Delta$, $B$, $C$) for the Selective State Space Model.
4. **Selective Scan**: The core SSM recurrence is applied across the sequence.
5. **Output Projection**: The SSM output is multiplicatively gated and projected back to the original model dimension.

## Convolutional Variants

The standard Mamba architecture uses a single depthwise causal convolution with a fixed kernel size. `lite-mamba` extends this with four distinct variants, accessible by importing the corresponding class:

### 1. TFPTCNMamba (Parallel TCN)
This is the default multi-branch implementation. Instead of a single convolution, it deploys **multiple parallel depthwise convolutions**, each with a different dilation rate (e.g., dilations 1, 2, 4, 8). 

* **Mechanism**: The output of these parallel branches is mixed together using learned softmax gating coefficients. 
* **Benefit**: This allows the model to dynamically attend to different receptive fields (short-term vs. long-term patterns) simultaneously.

### 2. TFSTCNMamba (Stacked TCN)
This variant arranges the different dilated depthwise convolutions **sequentially** rather than in parallel.

* **Mechanism**: The output of the dilation=1 branch feeds into the dilation=2 branch, which feeds into the dilation=4 branch, etc. There is no mixing gate.
* **Benefit**: Creates a deterministic, exponentially growing receptive field, similar to standard Temporal Convolutional Networks (TCNs). Useful for structural simplicity or debugging.

### 3. TFDPWCMamba (Depthwise + Pointwise)
This variant enhances the parallel branches of `TFPTCNMamba` by pairing each depthwise branch with a pointwise ($1 \times 1$) convolution *before* the gating mix.

* **Mechanism**: After the parallel dilated depthwise convolutions, a $1 \times 1$ convolution mixes the channels within each branch.
* **Benefit**: Adds richer cross-channel interactions at the convolutional stage without needing to stack additional deep layers.

### 4. TFBaselineMamba (Reference Baseline)
This is a functional equivalent to the original `state-spaces/mamba` architecture.

* **Mechanism**: Employs a single causal depthwise convolution.
* **Benefit**: Provides a reliable baseline to measure the improvements of the multi-branch variants above.
