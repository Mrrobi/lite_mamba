import pytest

from lite_mamba import TFBaselineMamba, TFDPWCMamba, TFPTCNMamba, TFSTCNMamba

tf = pytest.importorskip("tensorflow")


def _build_models(d_model=32, d_state=8, d_conv=3, conv_dilations=(1, 2, 4)):
    kwargs = {
        "d_model": d_model,
        "d_state": d_state,
        "d_conv": d_conv,
        "conv_dilations": conv_dilations,
    }
    return [
        TFBaselineMamba(d_model=d_model, d_state=d_state, d_conv=d_conv),
        TFPTCNMamba(**kwargs),
        TFSTCNMamba(**kwargs),
        TFDPWCMamba(**kwargs),
    ]


def test_tf_forward_shapes():
    tf.random.set_seed(0)
    x = tf.random.normal((2, 16, 32))
    for model in _build_models():
        y = model(x)
        assert y.shape == x.shape


def test_tf_step_matches_forward():
    tf.random.set_seed(0)
    batch, seqlen, d_model = 2, 12, 32
    x = tf.random.normal((batch, seqlen, d_model))
    for model in _build_models(d_model=d_model):
        full = model(x)
        conv_state, ssm_state = model.allocate_inference_cache(batch, seqlen)
        outs = []
        for t in range(seqlen):
            token = x[:, t : t + 1, :]
            out, conv_state, ssm_state = model.step(token, conv_state, ssm_state)
            outs.append(out)
        step_out = tf.concat(outs, axis=1)
        tf.debugging.assert_near(full, step_out, atol=1e-4, rtol=1e-4)
