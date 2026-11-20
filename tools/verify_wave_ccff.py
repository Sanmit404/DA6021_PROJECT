"""Mathematical & architectural verification suite for Wave-CCFF RT-DETR.

Verifies on CPU or GPU:
  1. Orthonormal 2D Haar DWT Parseval energy conservation:
       ||Z||_F^2 == ||LL||_F^2 + ||LH||_F^2 + ||HL||_F^2 + ||HH||_F^2
  2. Exact bijective reconstruction: IDWT(DWT(Z)) == Z
  3. Algebraic equivalence of baseline nearest-neighbor upsampling to
     degenerate zero-high-frequency Haar reconstruction:
       F.interpolate(P, scale_factor=2, mode='nearest') == IDWT(2P, 0, 0, 0)
  4. 100% pretrained state_dict key/shape compatibility between Baseline and Wave-CCFF
  5. Parameter count comparison and non-zero backward gradient flow through all Wave-CCFF modules
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn.functional as F

from src.core import YAMLConfig
from src.zoo.rtdetr.wave_hybrid_encoder import HaarDWT2D, HaarIDWT2D


def main():
    torch.manual_seed(42)
    dwt, idwt = HaarDWT2D(), HaarIDWT2D()

    # 1. Parseval & exact reconstruction check in fp64
    z = torch.randn(2, 256, 80, 80, dtype=torch.float64)
    ll, lh, hl, hh = dwt(z)
    energy_spatial = z.pow(2).sum().item()
    energy_spectral = (ll.pow(2).sum() + lh.pow(2).sum() + hl.pow(2).sum() + hh.pow(2).sum()).item()
    parseval_rel_err = abs(energy_spatial - energy_spectral) / energy_spatial

    rec_err = (z - idwt(ll, lh, hl, hh)).abs().max().item()

    p = torch.randn(2, 256, 40, 40, dtype=torch.float64)
    nearest_up = F.interpolate(p, scale_factor=2.0, mode='nearest')
    degenerate_haar_up = idwt(2.0 * p, torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p))
    nearest_equiv_err = (nearest_up - degenerate_haar_up).abs().max().item()

    assert parseval_rel_err < 1e-12, f'Parseval violation: {parseval_rel_err}'
    assert rec_err < 1e-12, f'IDWT reconstruction error: {rec_err}'
    assert nearest_equiv_err == 0.0, f'Nearest-upsample equivalence error: {nearest_equiv_err}'

    # 2. Build Baseline and Proposed Wave-CCFF models from configs
    cfg_base = YAMLConfig('configs/rtdetr/rtdetr_r18vd_visdrone_baseline.yml')
    model_base = cfg_base.model
    opt_base = cfg_base.optimizer

    cfg_wave = YAMLConfig('configs/rtdetr/rtdetr_r18vd_visdrone_wave_ccff.yml')
    model_wave = cfg_wave.model
    opt_wave = cfg_wave.optimizer

    params_base_m = sum(p.numel() for p in model_base.parameters()) / 1e6
    params_wave_m = sum(p.numel() for p in model_wave.parameters()) / 1e6
    delta_params_m = params_wave_m - params_base_m

    # 3. Verify 1:1 pretrained checkpoint key transferability
    base_sd = model_base.state_dict()
    wave_sd = model_wave.state_dict()
    matched = [k for k in base_sd if k in wave_sd and base_sd[k].shape == wave_sd[k].shape]
    assert len(matched) == len(base_sd), f'Matched {len(matched)}/{len(base_sd)} baseline keys'

    # 4. Verify forward + backward + non-zero gradients on all Wave-CCFF parameters
    x = torch.randn(2, 3, 320, 320)
    targets = [
        {'boxes': torch.tensor([[0.5, 0.5, 0.06, 0.06], [0.2, 0.3, 0.03, 0.04]]), 'labels': torch.tensor([0, 3])},
        {'boxes': torch.tensor([[0.7, 0.4, 0.05, 0.05]]), 'labels': torch.tensor([1])},
    ]
    outputs = model_wave(x, targets)
    loss_dict = cfg_wave.criterion(outputs, targets)
    total_loss = sum(loss_dict.values())
    total_loss.backward()

    wave_param_grads = {
        k: float(v.grad.abs().max().item())
        for k, v in model_wave.named_parameters()
        if ('wave_topdown' in k or 'wave_downsample' in k) and v.requires_grad
    }
    zero_grads = [k for k, g in wave_param_grads.items() if g == 0.0]
    assert len(zero_grads) == 0, f'Zero gradients detected in: {zero_grads}'

    # 5. Deploy-mode inference check
    model_wave_deploy = copy.deepcopy(model_wave).deploy() if 'copy' in globals() else model_wave.deploy()
    post_deploy = cfg_wave.postprocessor.deploy()
    with torch.no_grad():
        labels, boxes, scores = post_deploy(model_wave_deploy(x), torch.tensor([[320, 320], [320, 320]]))

    report = {
        'parseval_relative_error': parseval_rel_err,
        'idwt_max_reconstruction_error': rec_err,
        'nearest_vs_degenerate_haar_diff': nearest_equiv_err,
        'baseline_params_M': round(params_base_m, 4),
        'wave_ccff_params_M': round(params_wave_m, 4),
        'delta_params_M': round(delta_params_m, 4),
        'pretrained_keys_matched': f'{len(matched)}/{len(base_sd)}',
        'optimizer_groups_wave': len(opt_wave.param_groups),
        'wave_tensors_verified_nonzero_grad': len(wave_param_grads),
        'sample_train_loss': round(float(total_loss.item()), 4),
        'deploy_output_boxes_shape': list(boxes.shape),
    }
    print(json.dumps(report, indent=2))
    print('ALL WAVE-CCFF VERIFICATION CHECKS PASSED.')


if __name__ == '__main__':
    main()
