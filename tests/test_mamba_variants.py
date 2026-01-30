import torch

from lite_mamba import Mamba, PTCNMamba, STCNMamba, DPWCMamba


def _build_models(d_model=32, d_state=8, d_conv=3, conv_dilations=(1, 2, 4)):
    kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, conv_dilations=conv_dilations)
    return [
        Mamba(**kwargs),
        PTCNMamba(**kwargs),
        STCNMamba(**kwargs),
        DPWCMamba(**kwargs),
    ]


def test_forward_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32)
    for model in _build_models():
        y = model(x)
        assert y.shape == x.shape


def test_step_matches_forward():
    torch.manual_seed(0)
    batch, seqlen, d_model = 2, 12, 32
    x = torch.randn(batch, seqlen, d_model)
    for model in _build_models(d_model=d_model):
        model.eval()
        with torch.no_grad():
            full = model(x)
            conv_state, ssm_state = model.allocate_inference_cache(batch, seqlen)
            outs = []
            for t in range(seqlen):
                token = x[:, t : t + 1, :]
                out, conv_state, ssm_state = model.step(token, conv_state, ssm_state)
                outs.append(out)
            step_out = torch.cat(outs, dim=1)
        assert torch.allclose(full, step_out, atol=1e-4, rtol=1e-4)
