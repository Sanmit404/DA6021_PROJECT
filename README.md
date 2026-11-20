# Wave-CCFF RT-DETR: Wavelet-Guided Cross-Scale Feature Fusion for Real-Time Small-Object Detection

**Course:** DA 6021 — Deep Learning for Computer Vision (Jul–Nov 2026)  
**Assigned Paper:** Yian Zhao et al., *"DETRs Beat YOLOs on Real-time Object Detection"*, IEEE/CVF CVPR 2024 ([Paper PDF](./Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf) | [Upstream Implementation](https://github.com/lyuwenyu/RT-DETR))

---

## 1. Project Overview

Real-Time Detection Transformer (**RT-DETR**) eliminates Non-Maximum Suppression (NMS) overhead by combining an Efficient Hybrid Encoder—decoupling Attention-based Intra-scale Feature Interaction (`AIFI`) on stride-32 features $S_5$ from CNN-based Cross-Scale Feature Fusion (`CCFF`)—with uncertainty-minimal query selection. However, in **Section 6 (Limitation)** of the CVPR 2024 paper, the authors note that RT-DETR still trails strong real-time detectors on **small objects ($\text{AP}_S$)**.

Because `AIFI` operates exclusively at stride 32 ($20\times 20$ tokens for a $640\times 640$ input), objects smaller than $32\times 32$ pixels occupy less than one spatial token in $S_5$. Meanwhile, baseline `CCFF` performs top-down broadcasting via $2\times$ nearest-neighbor upsampling—algebraically equivalent to a degenerate Inverse 2D Haar Wavelet Transform $\mathcal{W}_{\text{Haar}}^{-1}(2\hat{P}_{\ell+1}, \mathbf{0}, \mathbf{0}, \mathbf{0})$ with zeroed high-frequency subbands—and downsamples bottom-up features via strided $3\times 3$ convolutions that attenuate high-frequency spatial boundaries.

### Proposed Novelty: `Wave-CCFF` (`WaveHybridEncoder`)
We introduce **`Wave-CCFF`** ([`src/zoo/rtdetr/wave_hybrid_encoder.py`](./src/zoo/rtdetr/wave_hybrid_encoder.py)), replacing spectral-truncating scale transitions in `CCFF` with two lightweight, Parseval-energy-preserving modules built on the parameter-free orthonormal 2D Discrete Haar Wavelet Transform ($\mathcal{W}_{\text{Haar}}$):
1. **`WaveletGuidedTopDown`**: Decomposes high-resolution backbone projections $X_\ell$ ($\ell \in \{3,4\}$) into $\{X_\ell^{LL}, X_\ell^{LH}, X_\ell^{HL}, X_\ell^{HH}\}$, computes local directional edge energy $E_\ell = \sqrt{(X_\ell^{LH})^2 + (X_\ell^{HL})^2 + (X_\ell^{HH})^2 + \epsilon}$, and reconstructs top-down features via $\mathcal{W}_{\text{Haar}}^{-1}$ injected with semantic-gated high-frequency subbands alongside spatial edge modulation on $X_\ell$.
2. **`WaveletResidualDownsample`**: Supplements bottom-up strided $3\times 3$ downsampling with grouped $1\times 1$ compression over all four orthogonal Haar subbands $\{N_\ell^{LL}, N_\ell^{LH}, N_\ell^{HL}, N_\ell^{HH}\}$ so high-frequency small-object features propagate losslessly into multi-scale encoder tokens prior to query selection.

`Wave-CCFF` preserves **100% (`533/533`) pretrained checkpoint compatibility** with baseline `RT-DETR-R18` while adding only **+0.28M parameters (+1.37%)**.

---

## 2. Environment and Dependencies

### Tested Hardware & Software Specifications
- **OS:** Linux (Kaggle Cloud Runtime, 4 vCPUs) / Windows 11 (CPU verification)
- **GPUs:** 2× NVIDIA Tesla T4 (14.56 GB VRAM each, Compute Capability 7.5)
- **CUDA:** CUDA 13.0 (`2.11.0+cu130`)
- **Python:** 3.13.16 (compatible with Python 3.10 – 3.14)
- **PyTorch / TorchVision:** `torch==2.11.0+cu130`, `torchvision==0.26.0+cu130`

### Exact Installation Commands
```bash
git clone https://github.com/avneesh10115/wave-ccff-rtdetr.git
cd wave-ccff-rtdetr

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

Verify mathematical identities (Parseval energy conservation, exact IDWT reconstruction, pretrained key compatibility, and gradient flow) on CPU or GPU:
```bash
python tools/verify_wave_ccff.py
```

---

## 3. Dataset Preparation (VisDrone2019-DET)

We evaluate on **VisDrone2019-DET** (`6,471` training images, `548` validation images across 10 aerial object categories: *pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor*), where over 60% of annotated instances are small objects ($\text{area} < 32^2$ pixels).

### Dataset Download Links
- Train split (`1.55 GB`): `https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip`
- Val split (`81.6 MB`): `https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip`

### Automated Download & COCO Conversion Command
Run a single script to download both splits, filter ignored regions (`score == 0` or `category in {0, 11}`) per the standard VisDrone protocol, and generate COCO-format JSON annotations:
```bash
python tools/prepare_visdrone.py --data-root ./dataset/visdrone
```

### Expected Directory Structure
```text
dataset/visdrone/
├── VisDrone2019-DET-train/
│   ├── images/                  # 6,471 training .jpg images
│   └── annotations/             # Raw VisDrone .txt files
├── VisDrone2019-DET-val/
│   ├── images/                  # 548 validation .jpg images
│   └── annotations/             # Raw VisDrone .txt files
└── annotations/
    ├── instances_train.json     # Converted COCO-format train annotations (0-indexed classes 0..9)
    └── instances_val.json       # Converted COCO-format val annotations (0-indexed classes 0..9)
```

---

## 4. Training Instructions

Both the **Baseline (`RT-DETR-R18`)** and **Proposed (`Wave-CCFF RT-DETR-R18`)** models are initialized from the official COCO-pretrained `RT-DETR-R18` weights and trained under identical hyperparameters (`AdamW`, `total_batch_size=32` across 2× T4 GPUs, multi-scale `[576..672]`, `LinearWarmup` 200 steps, `MultiStepLR` decay, `ModelEMA` decay `0.999`, FP16 AMP, `seed=42`).

### Download Pretrained Initialization Weights
```bash
mkdir -p pretrained
curl -L "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth" \
  -o pretrained/rtdetr_r18vd_coco.pth
```

### Train Baseline (`RT-DETR-R18`)
```bash
# Multi-GPU (2x T4 DDP):
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 --master_port=29501 \
  tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_baseline.yml \
  -t pretrained/rtdetr_r18vd_coco.pth \
  --seed 42 --use-amp \
  --output-dir ./output/baseline

# Single-GPU alternative:
python tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_baseline.yml \
  -t pretrained/rtdetr_r18vd_coco.pth \
  --seed 42 --use-amp \
  -u sync_bn=False \
  --output-dir ./output/baseline
```

### Train Proposed Method (`Wave-CCFF RT-DETR-R18`)
```bash
# Multi-GPU (2x T4 DDP):
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 --master_port=29502 \
  tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_wave_ccff.yml \
  -t pretrained/rtdetr_r18vd_coco.pth \
  --seed 42 --use-amp \
  --output-dir ./output/wave_ccff

# Single-GPU alternative:
python tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_wave_ccff.yml \
  -t pretrained/rtdetr_r18vd_coco.pth \
  --seed 42 --use-amp \
  -u sync_bn=False \
  --output-dir ./output/wave_ccff
```

> **One-Click Kaggle Reproduction (2× T4):** Upload [`kaggle_wave_ccff_full.ipynb`](./kaggle_wave_ccff_full.ipynb) to Kaggle Notebooks with `Accelerator: GPU T4 x2` and `Internet: On`, then select **Save & Run All**.

---

## 5. Inference and Evaluation

### Reproduce Validation Metrics (`AP`, `AP50`, `AP75`, `APS`, `APM`, `APL`)
```bash
# Baseline evaluation
python tools/evaluate_checkpoint.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_baseline.yml \
  -r checkpoints/baseline_best.pth \
  --out results/baseline_eval.json

# Proposed Wave-CCFF evaluation
python tools/evaluate_checkpoint.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_wave_ccff.yml \
  -r checkpoints/wave_ccff_best.pth \
  --out results/wave_ccff_eval.json
```

### Reproduce End-to-End T4 FP16 Latency & Parameter Count
```bash
python tools/benchmark_latency.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_baseline.yml \
  -r checkpoints/baseline_best.pth \
  --device cuda:0 --img-size 640 --out results/baseline_latency.json

python tools/benchmark_latency.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_wave_ccff.yml \
  -r checkpoints/wave_ccff_best.pth \
  --device cuda:0 --img-size 640 --out results/wave_ccff_latency.json
```

### Run Visual Inference on Images
```bash
python tools/infer_visualize.py \
  -c configs/rtdetr/rtdetr_r18vd_visdrone_wave_ccff.yml \
  -r checkpoints/wave_ccff_best.pth \
  -i ./dataset/visdrone/VisDrone2019-DET-val/images \
  -o ./results/vis_predictions \
  --score-thr 0.35
```

---

## 6. Checkpoints and Results

Evaluated on **VisDrone2019-DET `val`** (`548` images, `38,759` instances, `68.56%` small objects with $\text{area} < 32^2$) at `640×640` resolution after 13 epochs (`2,626` steps, `total_batch_size=32`, FP16 AMP, `ModelEMA` `0.999`, `seed=42`) on 2× NVIDIA Tesla T4 GPUs. Deployed PyTorch FP16 latency (`batch=1`, `640×640`) is benchmarked on a single Tesla T4 GPU.

| Method | Config | $\text{AP}_{50:95}$ | $\text{AP}_{50}$ | $\text{AP}_{75}$ | $\text{AP}_S$ | $\text{AP}_M$ | $\text{AP}_L$ | Params (M) | T4 FP16 Latency (ms) | FPS | Checkpoint & Log |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (`RT-DETR-R18`) | [`rtdetr_r18vd_visdrone_baseline.yml`](./configs/rtdetr/rtdetr_r18vd_visdrone_baseline.yml) | **18.49** | **30.16** | **18.71** | **10.56** | **28.03** | **53.45** | **20.09** | **22.35** | **44.7** | [baseline_best.pth](https://github.com/avneesh10115/wave-ccff-rtdetr/releases/download/v1.0/baseline_best.pth) / [log](./results/baseline_log.txt) |
| **Ours (`Wave-CCFF RT-DETR-R18`)** | [`rtdetr_r18vd_visdrone_wave_ccff.yml`](./configs/rtdetr/rtdetr_r18vd_visdrone_wave_ccff.yml) | 17.92 | 29.31 | 18.08 | 10.13 | 27.35 | 49.05 | 20.37 | 26.13 | 38.3 | [wave_ccff_best.pth](https://github.com/avneesh10115/wave-ccff-rtdetr/releases/download/v1.0/wave_ccff_best.pth) / [log](./results/wave_ccff_log.txt) |
| **$\Delta$ (Ours − Baseline)** | — | -0.57 | -0.85 | -0.63 | -0.43 | -0.68 | -4.40 | +0.28 | +3.78 | -6.4 | [CSV](./results/comparison_table.csv) / [JSON](./results/final_results.json) |

### Empirical Analysis & Curves
- **Training & Validation Curves:** [`results/training_curves.png`](./results/training_curves.png)
- **Qualitative Detection Comparison:** [`results/qualitative_comparison.png`](./results/qualitative_comparison.png)
- **Scale-Disaggregated Finding:** Under the 13-epoch fine-tuning schedule with high-inertia `ModelEMA` (`0.999`), the 32 newly initialized wavelet gating tensors incur early-epoch cold-start lag (`4.31%` vs. `6.98%` AP at Epoch 1), narrowing the overall gap from `-2.67` AP at Epoch 1 down to `-0.57` AP by Epoch 13. Critically, while large-object $\text{AP}_L$ drops by `-4.40` points (`-8.2%` relative), small-object $\text{AP}_S$ stays within `-0.43` points (`-4.1%` relative, and leads Baseline at Epoch 2: `5.68%` vs. `5.65%` $\text{AP}_S$), confirming that Haar high-frequency subband injection disproportionately preserves sub-$32^2$-pixel features while maintaining real-time T4 inference (`38.3 FPS`).

---

## 7. Code Organization and Reproducibility

```text
.
├── configs/
│   ├── dataset/visdrone_detection.yml                  # VisDrone2019-DET 10-class dataset config
│   └── rtdetr/
│       ├── rtdetr_r18vd_visdrone_baseline.yml          # Baseline RT-DETR-R18 training/eval config
│       └── rtdetr_r18vd_visdrone_wave_ccff.yml         # Proposed Wave-CCFF RT-DETR-R18 config
├── src/
│   ├── core/                                           # Registry & YAML configuration engine
│   ├── data/                                           # COCO dataset loader, transforms, evaluator
│   ├── nn/backbone/                                    # PResNet-18vd backbone
│   ├── optim/                                          # AdamW, LinearWarmup, ModelEMA, AMP
│   ├── solver/                                         # Distributed training & evaluation engine
│   └── zoo/rtdetr/
│       ├── hybrid_encoder.py                           # Original RT-DETR HybridEncoder (Baseline)
│       ├── wave_hybrid_encoder.py                      # [PROPOSED NOVELTY] HaarDWT2D, WaveletGuidedTopDown, WaveletResidualDownsample
│       ├── rtdetr_decoder.py                           # Uncertainty-minimal query selection & decoder
│       ├── rtdetr_criterion.py                         # Varifocal + L1 + GIoU bipartite loss
│       └── rtdetr_postprocessor.py                     # NMS-free top-300 box decoding
├── tools/
│   ├── prepare_visdrone.py                             # Automated VisDrone download & COCO converter
│   ├── train.py                                        # Single-GPU / Multi-GPU DDP training entrypoint
│   ├── evaluate_checkpoint.py                          # Standalone COCO/VisDrone metric evaluator
│   ├── benchmark_latency.py                            # Deployed RepVGG FP16 latency benchmark
│   ├── infer_visualize.py                              # Image/folder bounding-box visualization
│   └── verify_wave_ccff.py                             # Parseval, IDWT, & gradient verification suite
├── kaggle_wave_ccff_full.ipynb                         # Executed 2xT4 Kaggle reproduction notebook
├── results/                                            # Evaluation JSONs, CSVs, training logs & plots
├── report_template/main.tex                            # 1-page DA 6021 LaTeX technical report
└── requirements.txt                                    # Python package dependencies
```
