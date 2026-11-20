"""Wave-CCFF: Haar Wavelet High-Frequency Guided Cross-Scale Feature Fusion for RT-DETR.

Addresses the small-object (AP_S) bottleneck identified in Section 6 of
Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024):
  1. Orthonormal 2D Discrete Haar Wavelet Transform (HaarDWT2D / HaarIDWT2D)
     satisfying exact Parseval energy conservation:
       ||Z||_F^2 = ||Z_LL||_F^2 + ||Z_LH||_F^2 + ||Z_HL||_F^2 + ||Z_HH||_F^2.
  2. Top-down Wavelet-Guided Subband Upsampling & Spatial Gating (WaveletGuidedTopDown):
     replaces degenerate nearest-neighbor upsampling W_Haar^{-1}(2 P_hat, 0, 0, 0)
     with semantic-gated inverse Haar reconstruction using high-resolution directional
     subbands {LH, HL, HH} and modulates fine-scale features with local edge energy.
  3. Bottom-up Wavelet-Residual Downsampling (WaveletResidualDownsample):
     supplements strided 3x3 convolution with full-spectrum Haar subband compression
     to prevent high-frequency aliasing of sub-32x32 pixel objects.
"""

import copy
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_encoder import ConvNormLayer, CSPRepLayer, TransformerEncoder, TransformerEncoderLayer
from .utils import get_activation
from ...core import register


__all__ = ['WaveHybridEncoder', 'HaarDWT2D', 'HaarIDWT2D']


class HaarDWT2D(nn.Module):
    """Parameter-free orthonormal 2D Discrete Haar Wavelet Transform.

    Decomposes [B, C, 2H, 2W] -> (LL, LH, HL, HH), each of shape [B, C, H, W],
    using orthonormal 2x2 Haar kernels with exact Parseval energy preservation.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 + x01 - x10 - x11)
        hl = 0.5 * (x00 - x01 + x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """Parameter-free orthonormal 2D Inverse Discrete Haar Wavelet Transform.

    Reconstructs (LL, LH, HL, HH) of shape [B, C, H, W] -> [B, C, 2H, 2W].
    Note: HaarIDWT2D(2 * P, 0, 0, 0) is algebraically identical to
    F.interpolate(P, scale_factor=2, mode='nearest').
    """

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        x00 = 0.5 * (ll + lh + hl + hh)
        x01 = 0.5 * (ll + lh - hl - hh)
        x10 = 0.5 * (ll - lh + hl - hh)
        x11 = 0.5 * (ll - lh - hl + hh)

        top = torch.stack([x00, x01], dim=4).flatten(3, 4)
        bot = torch.stack([x10, x11], dim=4).flatten(3, 4)
        return torch.stack([top, bot], dim=3).flatten(2, 3)


class WaveletGuidedTopDown(nn.Module):
    """Top-down cross-scale fusion using Haar high-frequency subband injection & spatial gating."""

    def __init__(self, hidden_dim: int = 256, reduction: int = 4, act: str = 'silu', init_scale: float = 0.05):
        super().__init__()
        mid_dim = max(hidden_dim // reduction, 32)
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.context_bottleneck = ConvNormLayer(hidden_dim * 2, mid_dim, 1, 1, act=act)
        self.subband_gate = nn.Conv2d(mid_dim, hidden_dim, kernel_size=1, bias=True)
        self.fine_gate = nn.Conv2d(mid_dim, hidden_dim, kernel_size=1, bias=True)

        self.gamma_up = nn.Parameter(torch.full((1, hidden_dim, 1, 1), init_scale))
        self.gamma_fine = nn.Parameter(torch.full((1, hidden_dim, 1, 1), init_scale))

    def forward(self, feat_high: torch.Tensor, feat_low: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_h, orig_w = feat_low.shape[2:]
        _, lh, hl, hh = self.dwt(feat_low)

        # Compute directional high-frequency energy in fp32 to prevent fp16 underflow/overflow
        hf_energy = torch.sqrt(
            lh.float().pow(2) + hl.float().pow(2) + hh.float().pow(2) + 1e-6
        ).to(feat_low.dtype)

        if hf_energy.shape[2:] != feat_high.shape[2:]:
            feat_high_aligned = F.interpolate(feat_high, size=hf_energy.shape[2:], mode='nearest')
        else:
            feat_high_aligned = feat_high

        joint_ctx = self.context_bottleneck(torch.cat([hf_energy, feat_high_aligned], dim=1))
        g_sub = torch.tanh(self.subband_gate(joint_ctx))
        g_fine = torch.sigmoid(self.fine_gate(joint_ctx))

        scale_up = self.gamma_up.to(feat_low.dtype) * g_sub
        upsampled = self.idwt(
            2.0 * feat_high_aligned,
            scale_up * lh,
            scale_up * hl,
            scale_up * hh,
        )
        if upsampled.shape[2:] != (orig_h, orig_w):
            upsampled = upsampled[:, :, :orig_h, :orig_w]

        g_fine_up = F.interpolate(g_fine, size=(orig_h, orig_w), mode='nearest')
        gated_low = feat_low * (1.0 + self.gamma_fine.to(feat_low.dtype) * (g_fine_up - 0.5))
        return upsampled, gated_low


class WaveletResidualDownsample(nn.Module):
    """Bottom-up wavelet-preserving downsampling preserving {LL, LH, HL, HH} subband energy."""

    def __init__(self, hidden_dim: int = 256, act: str = 'silu', init_scale: float = 0.05):
        super().__init__()
        self.dwt = HaarDWT2D()
        groups = max(hidden_dim // 4, 1)
        self.subband_compress = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        self.beta_down = nn.Parameter(torch.full((1, hidden_dim, 1, 1), init_scale))

    def forward(self, feat_low: torch.Tensor, base_down: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.dwt(feat_low)
        # Group each channel's 4 orthogonal subbands contiguously: [B, 4*C, H/2, W/2]
        subbands = torch.stack([ll, lh, hl, hh], dim=2).flatten(1, 2)
        wave_feat = self.subband_compress(subbands)
        if wave_feat.shape[2:] != base_down.shape[2:]:
            wave_feat = F.interpolate(wave_feat, size=base_down.shape[2:], mode='bilinear', align_corners=False)
        return base_down + self.beta_down.to(base_down.dtype) * wave_feat


@register()
class WaveHybridEncoder(nn.Module):
    """RT-DETR Efficient Hybrid Encoder enhanced with Wave-CCFF."""

    __share__ = ['eval_spatial_size']

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act='silu',
        eval_spatial_size=None,
        version='v1',
        wave_reduction=4,
        wave_init_scale=0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # Keep exact attribute names of HybridEncoder so pretrained checkpoints load 1:1
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
            elif version == 'v2':
                proj = nn.Sequential(
                    OrderedDict([
                        ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                        ('norm', nn.BatchNorm2d(hidden_dim)),
                    ])
                )
            else:
                raise AttributeError(f'Unsupported version: {version}')
            self.input_proj.append(proj)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # Top-down FPN + Wave-CCFF high-frequency subband gating
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.wave_topdown = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
            self.wave_topdown.append(
                WaveletGuidedTopDown(hidden_dim, reduction=wave_reduction, act=act, init_scale=wave_init_scale)
            )

        # Bottom-up PAN + Wave-CCFF lossless Haar subband residual downsampling
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        self.wave_downsample = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
            self.wave_downsample.append(
                WaveletResidualDownsample(hidden_dim, act=act, init_scale=wave_init_scale)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                self.register_buffer(f'pos_embed{idx}', pos_embed, persistent=False)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Intra-scale interaction (AIFI on S5)
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                cached_pe = getattr(self, f'pos_embed{enc_ind}', None)
                if self.training or self.eval_spatial_size is None or cached_pe is None or cached_pe.shape[1] != h * w:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = cached_pe.to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # Top-down Wave-CCFF fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            stage_i = len(self.in_channels) - 1 - idx
            feat_high = self.lateral_convs[stage_i](inner_outs[0])
            feat_low = proj_feats[idx - 1]
            inner_outs[0] = feat_high

            upsample_feat, gated_low = self.wave_topdown[stage_i](feat_high, feat_low)
            inner_out = self.fpn_blocks[stage_i](torch.concat([upsample_feat, gated_low], dim=1))
            inner_outs.insert(0, inner_out)

        # Bottom-up Wave-CCFF fusion
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            base_down = self.downsample_convs[idx](feat_low)
            downsample_feat = self.wave_downsample[idx](feat_low, base_down)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs
