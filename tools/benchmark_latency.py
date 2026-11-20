"""End-to-end latency & parameter benchmark for RT-DETR Baseline vs Wave-CCFF.

Follows Section 3.2 of Zhao et al. (CVPR 2024):
  - Fuses RepVGG blocks via model.deploy() + postprocessor.deploy()
  - Benchmarks batch_size=1 at 640x640 resolution in FP16 on CUDA GPU (e.g., T4)
  - Uses CUDA event timing with warmup for accurate millisecond latency
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch

from src.core import YAMLConfig


class EndToEndDeployedDetector(torch.nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


@torch.no_grad()
def benchmark_model(cfg_path: str, checkpoint: str = None, device_str: str = 'cuda:0',
                    img_size: int = 640, warmup: int = 100, iters: int = 300, fp16: bool = True) -> dict:
    cfg = YAMLConfig(cfg_path)
    model = cfg.model

    # Parameter count BEFORE RepVGG structural re-parameterization fusion (training architecture params)
    total_params = sum(p.numel() for p in model.parameters())

    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        weights = state['ema']['module'] if 'ema' in state else state.get('model', state)
        model.load_state_dict(weights, strict=False)

    detector = EndToEndDeployedDetector(model, cfg.postprocessor)
    deployed_params = sum(p.numel() for p in detector.model.parameters())

    use_cuda = torch.cuda.is_available() and device_str.startswith('cuda')
    device = torch.device(device_str if use_cuda else 'cpu')
    detector = detector.to(device).eval()

    dummy_img = torch.randn(1, 3, img_size, img_size, device=device)
    orig_sizes = torch.tensor([[img_size, img_size]], device=device)

    use_amp_fp16 = bool(fp16 and use_cuda)

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        for _ in range(warmup):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_fp16):
                _ = detector(dummy_img, orig_sizes)
        torch.cuda.synchronize(device)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings_ms = []
        for _ in range(iters):
            starter.record()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_fp16):
                _ = detector(dummy_img, orig_sizes)
            ender.record()
            torch.cuda.synchronize(device)
            timings_ms.append(starter.elapsed_time(ender))

        timings_ms.sort()
        # Trim top/bottom 5% outliers for stable hardware reporting
        trim = max(int(0.05 * len(timings_ms)), 1)
        clean_ms = timings_ms[trim:-trim] if len(timings_ms) > 2 * trim else timings_ms
        mean_ms = sum(clean_ms) / len(clean_ms)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        for _ in range(min(warmup, 5)):
            _ = detector(dummy_img, orig_sizes)
        t0 = time.perf_counter()
        n_cpu = min(iters, 20)
        for _ in range(n_cpu):
            _ = detector(dummy_img, orig_sizes)
        mean_ms = (time.perf_counter() - t0) * 1000.0 / n_cpu
        gpu_name = 'CPU'

    return {
        'config': cfg_path,
        'device': gpu_name,
        'fp16': use_amp_fp16,
        'img_size': img_size,
        'params_M': round(total_params / 1e6, 2),
        'params_exact_M': round(total_params / 1e6, 4),
        'deployed_params_M': round(deployed_params / 1e6, 2),
        'latency_ms': round(mean_ms, 2),
        'fps': round(1000.0 / mean_ms, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    res = benchmark_model(
        args.config,
        checkpoint=args.checkpoint,
        device_str=args.device,
        img_size=args.img_size,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
