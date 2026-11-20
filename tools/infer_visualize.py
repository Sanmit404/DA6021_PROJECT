"""Run inference with trained RT-DETR / Wave-CCFF checkpoint on an image or folder."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from src.core import YAMLConfig


VISDRONE_CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor',
]

PALETTE = [
    '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231',
    '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
]


@torch.no_grad()
def run_inference(cfg_path: str, checkpoint_path: str, input_path: str, output_dir: str,
                  score_thr: float = 0.35, img_size: int = 640, device_str: str = None):
    cfg = YAMLConfig(cfg_path)
    if device_str is None:
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)

    model = cfg.model
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    weights = state['ema']['module'] if ('ema' in state and 'module' in state['ema']) else state.get('model', state)
    model.load_state_dict(weights, strict=False)
    model = model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()

    in_p = Path(input_path)
    img_paths = [in_p] if in_p.is_file() else sorted(
        [p for p in in_p.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    )
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    for img_file in img_paths:
        pil_img = Image.open(img_file).convert('RGB')
        orig_w, orig_h = pil_img.size
        resized = pil_img.resize((img_size, img_size), Image.BILINEAR)
        tensor_in = TF.to_tensor(resized).unsqueeze(0).to(device)
        orig_sizes = torch.tensor([[orig_w, orig_h]], device=device)

        outputs = model(tensor_in)
        res = postprocessor(outputs, orig_sizes)[0]

        draw = ImageDraw.Draw(pil_img)
        keep = res['scores'] >= score_thr
        boxes = res['boxes'][keep].cpu().tolist()
        labels = res['labels'][keep].cpu().tolist()
        scores = res['scores'][keep].cpu().tolist()

        for box, lab, sc in zip(boxes, labels, scores):
            color = PALETTE[int(lab) % len(PALETTE)]
            cls_name = VISDRONE_CLASSES[int(lab)] if int(lab) < len(VISDRONE_CLASSES) else str(lab)
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, max(0.0, box[1] - 12)), f'{cls_name} {sc:.2f}', fill=color)

        save_file = out_p / img_file.name
        pil_img.save(save_file)
        print(f'Saved {save_file} ({len(boxes)} detections >= {score_thr})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--checkpoint', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='./outputs/vis', help='Output folder')
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    run_inference(args.config, args.checkpoint, args.input, args.output,
                  score_thr=args.score_thr, device_str=args.device)


if __name__ == '__main__':
    main()
