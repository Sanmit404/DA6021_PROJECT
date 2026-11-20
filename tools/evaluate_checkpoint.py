"""Standalone COCO / VisDrone evaluation script producing structured JSON metrics.

Reproduces exact AP, AP50, AP75, APS, APM, APL (%) from a trained checkpoint.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch

from src.core import YAMLConfig, yaml_utils
from src.solver.det_engine import evaluate


def evaluate_model(cfg_path: str, checkpoint_path: str, device_str: str = None, updates=None) -> dict:
    update_dict = yaml_utils.parse_cli(updates) if updates else {}
    cfg = YAMLConfig(cfg_path, **update_dict)

    if device_str is None:
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    model = cfg.model
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'ema' in state and isinstance(state['ema'], dict) and 'module' in state['ema']:
        weights = state['ema']['module']
        weight_source = 'ema'
    elif 'model' in state:
        weights = state['model']
        weight_source = 'model'
    else:
        weights = state
        weight_source = 'raw'

    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()
    criterion = cfg.criterion.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()

    val_loader = cfg.val_dataloader
    evaluator = cfg.evaluator

    stats, _ = evaluate(model, criterion, postprocessor, val_loader, evaluator, device)
    bbox = stats['coco_eval_bbox']

    metrics = {
        'config': cfg_path,
        'checkpoint': checkpoint_path,
        'weight_source': weight_source,
        'AP': round(bbox[0] * 100.0, 2),
        'AP50': round(bbox[1] * 100.0, 2),
        'AP75': round(bbox[2] * 100.0, 2),
        'APS': round(bbox[3] * 100.0, 2),
        'APM': round(bbox[4] * 100.0, 2),
        'APL': round(bbox[5] * 100.0, 2),
        'AR100': round(bbox[8] * 100.0, 2),
        'raw_coco_stats': [round(x * 100.0, 3) for x in bbox],
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument('-u', '--update', nargs='+', default=None)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    metrics = evaluate_model(args.config, args.checkpoint, device_str=args.device, updates=args.update)
    print(json.dumps(metrics, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
