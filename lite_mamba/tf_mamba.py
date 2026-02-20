import math

import tensorflow as tf


def selective_scan_tf(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """TensorFlow reference selective scan (XLA-compatible via tf.scan).

    Shapes:
      u: (B, L, D)
      delta: (B, L, D)
      A: (D, N)
      B: (B, L, N)
      C: (B, L, N)
      D: (D,)
      z: (B, L, D)
    """
    dtype_in = u.dtype
    u = tf.cast(u, tf.float32)
    delta = tf.cast(delta, tf.float32)
    A = tf.cast(A, tf.float32)
    B = tf.cast(B, tf.float32)
    C = tf.cast(C, tf.float32)
    if D is not None:
        D = tf.cast(D, tf.float32)
    if z is not None:
        z = tf.cast(z, tf.float32)
    if delta_bias is not None:
        delta = delta + tf.reshape(tf.cast(delta_bias, tf.float32), [1, 1, -1])
    if delta_softplus:
        delta = tf.nn.softplus(delta)

    batch = tf.shape(u)[0]
    dim = tf.shape(u)[2]
    d_state = tf.shape(A)[1]

    deltaA = tf.exp(tf.einsum("bld,dn->bldn", delta, A))  # (B, L, D, N)
    deltaB_u = tf.einsum("bld,bln,bld->bldn", delta, B, u)  # (B, L, D, N)

    # Transpose to (L, B, D, N) / (L, B, N) so tf.scan iterates over the
    # sequence axis (axis-0).  tf.scan does not use TensorArray internally
    # and is fully XLA-compilable.
    deltaA_t = tf.transpose(deltaA, perm=[1, 0, 2, 3])  # (L, B, D, N)
    deltaBu_t = tf.transpose(deltaB_u, perm=[1, 0, 2, 3])  # (L, B, D, N)
    C_t = tf.transpose(C, perm=[1, 0, 2])  # (L, B, N)

    x0 = tf.zeros([batch, dim, d_state], dtype=tf.float32)

    # tf.scan signature: fn(accumulator, elem) -> new_accumulator
    # The accumulator is stacked across all steps automatically.
    # We compute y from the stacked x afterwards (avoids needing two outputs).
    def scan_fn(x_prev, elems):
        dA_i, dBu_i = elems  # (B,D,N), (B,D,N)
        x_new = dA_i * x_prev + dBu_i  # (B, D, N)
        return x_new

    # x_all: (L, B, D, N) — SSM hidden state at every timestep
    x_all = tf.scan(
        fn=scan_fn,
        elems=(deltaA_t, deltaBu_t),
        initializer=x0,
    )

    # Compute y for all timesteps in one vectorised einsum:
    # x_all: (L, B, D, N), C_t: (L, B, N) -> ys: (L, B, D)
    ys = tf.einsum("lbdn,lbn->lbd", x_all, C_t)

    y = tf.transpose(ys, perm=[1, 0, 2])  # (L, B, D) -> (B, L, D)
    out = y if D is None else y + u * tf.reshape(D, [1, 1, -1])
    if z is not None:
        out = out * tf.nn.silu(z)
    out = tf.cast(out, dtype_in)
    if return_last_state:
        return out, x_all[-1]  # last SSM state: (B, D, N)
    return out


class TFMamba(tf.keras.layers.Layer):
    """TensorFlow Mamba block with parallel dilated depthwise causal conv branches."""

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
        stacked_convs=False,
        pointwise=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_dilations = tuple(conv_dilations)
        self.num_conv_branches = len(self.conv_dilations)
        self.conv_state_lens = [(self.d_conv - 1) * d + 1 for d in self.conv_dilations]
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.bias = bias
        self.layer_idx = layer_idx
        self.stacked_convs = stacked_convs
        self.pointwise = pointwise

        self.in_proj = tf.keras.layers.Dense(self.d_inner * 2, use_bias=self.bias)
        self.x_proj = tf.keras.layers.Dense(self.dt_rank + self.d_state * 2, use_bias=False)
        self.dt_proj = tf.keras.layers.Dense(self.d_inner, use_bias=True)
        self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=self.bias)

        if self.num_conv_branches > 1 and not self.stacked_convs:
            self.conv_gates = self.add_weight(
                name="conv_gates",
                shape=(self.num_conv_branches,),
                initializer=tf.keras.initializers.Ones(),
                trainable=True,
            )
        else:
            self.conv_gates = None

        self.dw_kernels = []
        self.dw_biases = []
        self.pw_layers = []
        for i in range(self.num_conv_branches):
            self.dw_kernels.append(
                self.add_weight(
                    name=f"dw_kernel_{i}",
                    shape=(self.d_conv, self.d_inner),
                    initializer=tf.keras.initializers.GlorotUniform(),
                    trainable=True,
                )
            )
            if self.conv_bias:
                self.dw_biases.append(
                    self.add_weight(
                        name=f"dw_bias_{i}",
                        shape=(self.d_inner,),
                        initializer=tf.keras.initializers.Zeros(),
                        trainable=True,
                    )
                )
            else:
                self.dw_biases.append(None)
            if self.pointwise:
                self.pw_layers.append(tf.keras.layers.Dense(self.d_inner, use_bias=True))

        a = tf.cast(tf.range(1, self.d_state + 1), tf.float32)
        a = tf.tile(tf.expand_dims(a, 0), [self.d_inner, 1])
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.d_inner, self.d_state),
            initializer=tf.constant_initializer(tf.math.log(a).numpy()),
            trainable=True,
        )
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )

    def build(self, input_shape):
        self.in_proj.build((None, None, self.d_model))
        self.x_proj.build((None, None, self.d_inner))
        self.dt_proj.build((None, None, self.dt_rank))
        self.out_proj.build((None, None, self.d_inner))
        if self.pointwise:
            for pw in self.pw_layers:
                pw.build((None, None, self.d_inner))

        dt_init_std = (self.dt_rank**-0.5) * self.dt_scale
        if self.dt_init == "constant":
            kernel = tf.fill([self.dt_rank, self.d_inner], tf.cast(dt_init_std, tf.float32))
        elif self.dt_init == "random":
            kernel = tf.random.uniform(
                [self.dt_rank, self.d_inner],
                minval=-dt_init_std,
                maxval=dt_init_std,
                dtype=tf.float32,
            )
        else:
            raise NotImplementedError(f"Unsupported dt_init: {self.dt_init}")
        self.dt_proj.kernel.assign(kernel)

        log_min = math.log(self.dt_min)
        log_max = math.log(self.dt_max)
        dt = tf.exp(
            tf.random.uniform([self.d_inner], minval=log_min, maxval=log_max, dtype=tf.float32)
        )
        dt = tf.maximum(dt, self.dt_init_floor)
        inv_dt = dt + tf.math.log(-tf.math.expm1(-dt))
        self.dt_proj.bias.assign(inv_dt)
        super().build(input_shape)

    @staticmethod
    def _causal_depthwise_conv1d(x, kernel, bias=None, dilation=1):
        # x: (B, L, D), kernel: (K, D)
        k = kernel.shape[0]
        pad_left = dilation * (k - 1)
        xpad = tf.pad(x, [[0, 0], [pad_left, 0], [0, 0]])
        seqlen = tf.shape(x)[1]
        taps = [xpad[:, i * dilation : i * dilation + seqlen, :] for i in range(k)]
        stacked = tf.stack(taps, axis=2)  # (B, L, K, D)
        y = tf.reduce_sum(stacked * tf.reshape(kernel, [1, 1, k, -1]), axis=2)
        if bias is not None:
            y = y + tf.reshape(bias, [1, 1, -1])
        return tf.nn.silu(y)

    def _run_branch(self, x, branch_idx):
        y = self._causal_depthwise_conv1d(
            x,
            kernel=self.dw_kernels[branch_idx],
            bias=self.dw_biases[branch_idx],
            dilation=self.conv_dilations[branch_idx],
        )
        if self.pointwise:
            y = self.pw_layers[branch_idx](y)
        return y

    def call(self, hidden_states, training=None):
        # hidden_states: (B, L, D)
        xz = self.in_proj(hidden_states)
        x, z = tf.split(xz, num_or_size_splits=2, axis=-1)
        A = -tf.exp(tf.cast(self.A_log, tf.float32))

        if self.stacked_convs:
            for i in range(self.num_conv_branches):
                x = self._run_branch(x, i)
        else:
            conv_outputs = [self._run_branch(x, i) for i in range(self.num_conv_branches)]
            if self.conv_gates is None:
                x = conv_outputs[0]
            else:
                gate = tf.nn.softmax(self.conv_gates, axis=0)
                x = tf.add_n([gate[i] * conv_outputs[i] for i in range(self.num_conv_branches)])

        x_dbl = self.x_proj(x)
        dt, B, C = tf.split(
            x_dbl,
            num_or_size_splits=[self.dt_rank, self.d_state, self.d_state],
            axis=-1,
        )

        # Match PyTorch path: apply dt_proj weights here, then add bias/softplus inside scan.
        dt = tf.einsum("blr,rd->bld", dt, self.dt_proj.kernel)

        y = selective_scan_tf(
            x,
            dt,
            A,
            B,
            C,
            self.D,
            z=z,
            delta_bias=self.dt_proj.bias,
            delta_softplus=True,
            return_last_state=False,
        )
        return self.out_proj(y)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=tf.float32):
        conv_state = [
            tf.zeros([batch_size, state_len, self.d_inner], dtype=dtype)
            for state_len in self.conv_state_lens
        ]
        ssm_state = tf.zeros([batch_size, self.d_inner, self.d_state], dtype=dtype)
        return conv_state, ssm_state

    def step(self, hidden_states, conv_state, ssm_state):
        # hidden_states: (B, 1, D)
        xz = self.in_proj(hidden_states[:, 0, :])
        x, z = tf.split(xz, num_or_size_splits=2, axis=-1)
        A = -tf.exp(tf.cast(self.A_log, tf.float32))
        dtype = hidden_states.dtype

        def branch_step(x_in, state, kernel, bias, dilation, pw_layer):
            x_state = tf.concat([state[:, 1:, :], tf.expand_dims(x_in, axis=1)], axis=1)
            k = kernel.shape[0]
            idx = tf.range(k - 1, -1, -1) * dilation
            pos = tf.shape(x_state)[1] - 1 - idx
            values = tf.gather(x_state, pos, axis=1)  # (B, K, D), oldest->newest
            y = tf.reduce_sum(values * tf.reshape(kernel, [1, k, self.d_inner]), axis=1)
            if bias is not None:
                y = y + bias
            y = tf.nn.silu(y)
            if pw_layer is not None:
                y = pw_layer(y)
            return y, x_state

        if self.stacked_convs:
            new_states = []
            for i in range(self.num_conv_branches):
                pw = self.pw_layers[i] if self.pointwise else None
                x, new_state = branch_step(
                    x,
                    conv_state[i],
                    self.dw_kernels[i],
                    self.dw_biases[i],
                    self.conv_dilations[i],
                    pw,
                )
                new_states.append(new_state)
        else:
            branch_outputs = []
            new_states = []
            for i in range(self.num_conv_branches):
                pw = self.pw_layers[i] if self.pointwise else None
                xi, new_state = branch_step(
                    x,
                    conv_state[i],
                    self.dw_kernels[i],
                    self.dw_biases[i],
                    self.conv_dilations[i],
                    pw,
                )
                branch_outputs.append(xi)
                new_states.append(new_state)
            if self.conv_gates is None:
                x = branch_outputs[0]
            else:
                gate = tf.nn.softmax(self.conv_gates, axis=0)
                x = tf.add_n([gate[i] * branch_outputs[i] for i in range(self.num_conv_branches)])

        x_db = self.x_proj(x)
        dt, B, C = tf.split(
            x_db,
            num_or_size_splits=[self.dt_rank, self.d_state, self.d_state],
            axis=-1,
        )
        dt = tf.einsum("br,rd->bd", dt, self.dt_proj.kernel)
        dt = tf.nn.softplus(dt + self.dt_proj.bias)
        dA = tf.exp(tf.einsum("bd,dn->bdn", dt, A))
        dB = tf.einsum("bd,bn->bdn", dt, B)
        new_ssm_state = ssm_state * dA + tf.expand_dims(x, axis=-1) * dB
        y = tf.einsum("bdn,bn->bd", tf.cast(new_ssm_state, dtype), C)
        y = y + tf.cast(self.D, dtype) * x
        y = y * tf.nn.silu(z)
        out = self.out_proj(y)
        return tf.expand_dims(out, axis=1), new_states, new_ssm_state


class TFBaselineMamba(TFMamba):
    """Single-branch TensorFlow baseline Mamba."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("conv_dilations", None)
        super().__init__(*args, conv_dilations=(1,), **kwargs)


class TFPTCNMamba(TFMamba):
    """Parallel TCN branches (TensorFlow variant)."""


class TFSTCNMamba(TFMamba):
    """Stacked TCN branches (TensorFlow variant)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, stacked_convs=True, **kwargs)


class TFDPWCMamba(TFMamba):
    """Depthwise + pointwise branch variant (TensorFlow)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, pointwise=True, **kwargs)


def tf_baseline_mamba(*args, **kwargs):
    return TFBaselineMamba(*args, **kwargs)
