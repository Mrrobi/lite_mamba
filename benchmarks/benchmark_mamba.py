import argparse
import time

import torch

from lite_mamba import PTCNMamba, STCNMamba, DPWCMamba


def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_one(model, x, iters, warmup):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        _sync_if_cuda(x.device)
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        _sync_if_cuda(x.device)
        end = time.perf_counter()
    return (end - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--conv-dilations", type=str, default="1,2,4,8")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    conv_dilations = tuple(int(x) for x in args.conv_dilations.split(",") if x)

    x = torch.randn(args.batch, args.seqlen, args.d_model, device=device)
    kwargs = dict(
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        conv_dilations=conv_dilations,
        device=device,
    )
    models = [
        ("PTCNMamba", PTCNMamba(**kwargs)),
        ("STCNMamba", STCNMamba(**kwargs)),
        ("DPWCMamba", DPWCMamba(**kwargs)),
    ]

    print(
        f"device={device} batch={args.batch} seqlen={args.seqlen} d_model={args.d_model} "
        f"d_state={args.d_state} d_conv={args.d_conv} conv_dilations={conv_dilations}"
    )
    if args.smoke:
        for name, model in models:
            y = model(x)
            print(f"{name:10s}  output_shape={tuple(y.shape)}")
        return
    for name, model in models:
        ms = _bench_one(model, x, args.iters, args.warmup) * 1000.0
        print(f"{name:10s}  {ms:8.3f} ms/iter")


if __name__ == "__main__":
    main()
