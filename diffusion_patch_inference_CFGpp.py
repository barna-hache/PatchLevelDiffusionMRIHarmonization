from tqdm import tqdm
import numpy as np
import os
import random
import math
from typing import Tuple, Sequence, Optional
import warnings
import nibabel as nib
import argparse
import time
warnings.filterwarnings("ignore", message=".*torchio.*SubjectsLoader.*")

import gc
import torch
import torch.nn as nn
from torch.nn import functional as F
from accelerate import Accelerator
import torchio as tio
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.utils.checkpoint as cp
from diffusers import DDIMScheduler
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# dataset -> id map
ds2id = {
    'aibl_gamma_-0.1': 1, 'aibl_gamma_-0.2': 2, 'aibl_gamma_-0.3': 3, 'aibl_gamma_-0.4': 4,
    'aibl_gamma_0': 5, 'aibl_gamma_0.1': 6, 'aibl_gamma_0.2': 7, 'aibl_gamma_0.3': 8, 'aibl_gamma_0.4': 9,
    'hcp_gamma_-0.1': 10, 'hcp_gamma_-0.2': 11, 'hcp_gamma_-0.3': 12, 'hcp_gamma_-0.4': 13,
    'hcp_gamma_0': 14, 'hcp_gamma_0.1': 15, 'hcp_gamma_0.2': 16, 'hcp_gamma_0.3': 17, 'hcp_gamma_0.4': 18,
    'icbm-ACS_gamma_-0.1': 19, 'icbm-ACS_gamma_-0.2': 20, 'icbm-ACS_gamma_-0.3': 21, 'icbm-ACS_gamma_-0.4': 22,
    'icbm-ACS_gamma_0': 23, 'icbm-ACS_gamma_0.1': 24, 'icbm-ACS_gamma_0.2': 25, 'icbm-ACS_gamma_0.3': 26, 'icbm-ACS_gamma_0.4': 27,
    'icbm-sonata_gamma_-0.1': 28, 'icbm-sonata_gamma_-0.2': 29, 'icbm-sonata_gamma_-0.3': 30, 'icbm-sonata_gamma_-0.4': 31,
    'icbm-sonata_gamma_0': 32, 'icbm-sonata_gamma_0.1': 33, 'icbm-sonata_gamma_0.2': 34, 'icbm-sonata_gamma_0.3': 35, 'icbm-sonata_gamma_0.4': 36,
    'ixi-guys_gamma_-0.1': 37, 'ixi-guys_gamma_-0.2': 38, 'ixi-guys_gamma_-0.3': 39, 'ixi-guys_gamma_-0.4': 40,
    'ixi-guys_gamma_0': 41, 'ixi-guys_gamma_0.1': 42, 'ixi-guys_gamma_0.2': 43, 'ixi-guys_gamma_0.3': 44, 'ixi-guys_gamma_0.4': 45,
    'ixi-hh_gamma_-0.1': 46, 'ixi-hh_gamma_-0.2': 47, 'ixi-hh_gamma_-0.3': 48, 'ixi-hh_gamma_-0.4': 49,
    'ixi-hh_gamma_0': 50, 'ixi-hh_gamma_0.1': 51, 'ixi-hh_gamma_0.2': 52, 'ixi-hh_gamma_0.3': 53, 'ixi-hh_gamma_0.4': 54,
    'nmorph_gamma_-0.1': 55, 'nmorph_gamma_-0.2': 56, 'nmorph_gamma_-0.3': 57, 'nmorph_gamma_-0.4': 58,
    'nmorph_gamma_0': 59, 'nmorph_gamma_0.1': 60, 'nmorph_gamma_0.2': 61, 'nmorph_gamma_0.3': 62, 'nmorph_gamma_0.4': 63,
    'oas-bio_gamma_-0.1': 64, 'oas-bio_gamma_-0.2': 65, 'oas-bio_gamma_-0.3': 66, 'oas-bio_gamma_-0.4': 67,
    'oas-bio_gamma_0': 68, 'oas-bio_gamma_0.1': 69, 'oas-bio_gamma_0.2': 70, 'oas-bio_gamma_0.3': 71, 'oas-bio_gamma_0.4': 72,
    'oas-trio_gamma_-0.1': 73, 'oas-trio_gamma_-0.2': 74, 'oas-trio_gamma_-0.3': 75, 'oas-trio_gamma_-0.4': 76,
    'oas-trio_gamma_0': 77, 'oas-trio_gamma_0.1': 78, 'oas-trio_gamma_0.2': 79, 'oas-trio_gamma_0.3': 80, 'oas-trio_gamma_0.4': 81,
    'sald_gamma_-0.1': 82, 'sald_gamma_-0.2': 83, 'sald_gamma_-0.3': 84, 'sald_gamma_-0.4': 85,
    'sald_gamma_0': 86, 'sald_gamma_0.1': 87, 'sald_gamma_0.2': 88, 'sald_gamma_0.3': 89, 'sald_gamma_0.4': 90
}

# ----------------------
# Anatomy / image ops
# ----------------------

def _gaussian_kernel_3d(sigma: float, device: torch.device, dtype=torch.float32):
    if sigma <= 0:
        k = torch.zeros((1, 1, 1, 1, 1), device=device, dtype=dtype)
        k[0, 0, 0, 0, 0] = 1.0
        return k
    radius = int(max(1, torch.ceil(3 * torch.tensor(sigma)).item()))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    return g3d.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

def _pad_for_conv(kernel):
    kD, kH, kW = kernel.shape[-3:]
    return (kW // 2, kW // 2, kH // 2, kH // 2, kD // 2, kD // 2)

def _conv3d_same(x, kernel):
    pad = _pad_for_conv(kernel)
    x_p = F.pad(x, pad, mode='replicate')
    return F.conv3d(x_p, kernel)

def gaussian_blur3d(x, sigma: float):
    if sigma <= 0:
        return x
    kernel = _gaussian_kernel_3d(sigma, device=x.device, dtype=x.dtype)
    return _conv3d_same(x, kernel)

def _sobel_kernels_3d(device, dtype=torch.float32):
    kx = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    ky = torch.zeros_like(kx)
    kz = torch.zeros_like(kx)
    # central difference kernels
    kx[0, 0, 1, 1, 0] = -1.0; kx[0, 0, 1, 1, 2] = 1.0
    ky[0, 0, 1, 0, 1] = -1.0; ky[0, 0, 1, 2, 1] = 1.0
    kz[0, 0, 0, 1, 1] = -1.0; kz[0, 0, 2, 1, 1] = 1.0
    return kx, ky, kz

def gradient_magnitude_torch(x, sigma=0.0):
    if sigma > 0:
        x = gaussian_blur3d(x, sigma)
    kx, ky, kz = _sobel_kernels_3d(x.device, dtype=x.dtype)
    pad = _pad_for_conv(kx)
    xpad = F.pad(x, pad, mode='replicate')
    dx = F.conv3d(xpad, kx)
    dy = F.conv3d(xpad, ky)
    dz = F.conv3d(xpad, kz)
    return torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)

def _global_quantile(tensor, q, max_samples=2_000_000):
    """
    Approximate quantile computation using deterministic strided subsampling when tensor is large.
    """
    flat = tensor.view(-1)
    n = flat.numel()
    if n <= max_samples:
        return torch.quantile(flat, q)
    step = int(n // max_samples) + 1
    sample = flat[::step]
    return torch.quantile(sample, q)

def make_structural_anatomy_map(batch_imgs: torch.Tensor,
                                grad_sigmas=(0.5, 2.0),
                                hf_sigma=1.0,
                                smooth_sigma=1.0,
                                normalize_percentiles=(1.0, 99.0)):
    """
    Create a single-channel 3D anatomy map of the same size as input.
    Args:
        batch_imgs: (B,1,D,H,W)
    """
    assert batch_imgs.ndim == 5 and batch_imgs.shape[1] == 1
    device, dtype = batch_imgs.device, batch_imgs.dtype

    # multi-scale gradient magnitudes
    g1 = gradient_magnitude_torch(batch_imgs, sigma=grad_sigmas[0])
    g2 = gradient_magnitude_torch(batch_imgs, sigma=grad_sigmas[1])

    # high-frequency map |I - G_sigma(I)|
    blurred = gaussian_blur3d(batch_imgs, sigma=hf_sigma)
    hf = torch.abs(batch_imgs - blurred)

    combined = 0.5 * g1 + 0.3 * g2 + 0.2 * hf

    # robust normalization to [-1, 1] using global percentiles
    p1, p99 = normalize_percentiles
    lo = _global_quantile(combined, p1 / 100.0)
    hi = _global_quantile(combined, p99 / 100.0)
    normed = (combined - lo) / (hi - lo + 1e-6)
    normed = normed.clamp(0, 1) * 2 - 1

    if smooth_sigma > 0:
        normed = gaussian_blur3d(normed, sigma=smooth_sigma)

    return normed

# ----------------------
# Utilities
# ----------------------

def make_blending_window(patch_size):
    # separable 3D Hann window -> shape (pd,ph,pw)
    wx = torch.hann_window(patch_size[0], periodic=False)
    wy = torch.hann_window(patch_size[1], periodic=False)
    wz = torch.hann_window(patch_size[2], periodic=False)
    w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    return w.float()

def compute_tight_crop_coords(volume):
    """
    Return zmin,zmax,ymin,ymax,xmin,xmax (inclusive).
    volume: (1,1,D,H,W)
    """
    v = volume[0, 0]
    mask3 = ~torch.isclose(v, v.min(), atol=1e-6)
    zs, ys, xs = torch.where(mask3)
    if zs.numel() == 0:
        D, H, W = v.shape
        return 0, D - 1, 0, H - 1, 0, W - 1
    zmin, zmax = int(zs.min().item()), int(zs.max().item())
    ymin, ymax = int(ys.min().item()), int(ys.max().item())
    xmin, xmax = int(xs.min().item()), int(xs.max().item())
    return zmin, zmax, ymin, ymax, xmin, xmax

def compute_positions_no_pad(L, p):
    """
    Compute integer positions [pos0, ..., posN] so:
      - first = 0, last = L-p
      - patches size = p
      - average stride <= p/2 (>=50% overlap)
      - full coverage without padding
    Approach: choose minimal N satisfying stride <= p/2 then round linspace positions.
    """
    if L <= p:
        return np.array([0], dtype=int)
    min_N = int(np.ceil((L - p) / (p / 2.0))) + 1
    if min_N < 2:
        min_N = 2
    pos = np.round(np.linspace(0, L - p, min_N)).astype(int)
    pos = np.unique(pos)
    if pos.size == 1 and L - p > 0:
        pos = np.array([0, L - p], dtype=int)
    pos = np.clip(pos, 0, L - p)
    return pos

# ----------------------
# Diffusion helpers and model
# ----------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)

class Conv3dZeroInit(nn.Conv3d):
    """Conv3d with zero init option for residual projection (placeholder)."""
    pass

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.nin_shortcut = None
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        if self.time_mlp is not None and t_emb is not None:
            # t_emb shape: (B, dim)
            t = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            h = h + t
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h

class MultiHeadAttention3D(nn.Module):
    def __init__(self, dim, num_heads, head_dim, cross_dim=None):
        """
        Multi-head attention operating on flattened spatial tokens.
        cross_dim: if provided, keys/values come from context of that dim.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim if cross_dim is None else cross_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim if cross_dim is None else cross_dim, self.inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim))

    def forward(self, x, context: Optional[torch.Tensor] = None):
        b, n, _ = x.shape
        context = x if context is None else context
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.inner_dim)
        return self.to_out(out)

class AttentionBlock3D(nn.Module):
    def __init__(self, channels, num_heads, head_dim, cross_attention_dim: Optional[int] = None):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.proj_in = nn.Conv3d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)
        self.mha = MultiHeadAttention3D(dim=channels, num_heads=num_heads, head_dim=head_dim, cross_dim=cross_attention_dim)
        self.cross_attention_dim = cross_attention_dim

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        b, c, d, h, w = x.shape
        h_in = self.norm(x)
        h_in = F.silu(h_in)
        h_in = self.proj_in(h_in)
        h_flat = h_in.view(b, c, d * h * w).permute(0, 2, 1)  # (B, N, C)
        if context is not None:
            attn_out = self.mha(h_flat, context)
        else:
            attn_out = self.mha(h_flat, None)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, d, h, w)
        out = self.proj_out(attn_out)
        return x + out

class Downsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)

class Upsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose3d(channels, channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)

class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, use_attn=False, num_heads=1, head_dim=32, cross_attention_dim=None):
        super().__init__()
        self.res1 = ResidualBlock3D(in_ch, out_ch, time_emb_dim=time_emb_dim)
        self.attn = AttentionBlock3D(out_ch, num_heads, head_dim, cross_attention_dim) if use_attn else None
        self.res2 = ResidualBlock3D(out_ch, out_ch, time_emb_dim=time_emb_dim)
    def forward(self, x, t_emb=None, context=None):
        x = self.res1(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, context)
        x = self.res2(x, t_emb)
        return x

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, use_attn=False, num_heads=1, head_dim=32, cross_attention_dim=None):
        super().__init__()
        # in_ch is concatenated channels from skip + current
        self.res1 = ResidualBlock3D(in_ch, out_ch, time_emb_dim=time_emb_dim)
        self.attn = AttentionBlock3D(out_ch, num_heads, head_dim, cross_attention_dim) if use_attn else None
        self.res2 = ResidualBlock3D(out_ch, out_ch, time_emb_dim=time_emb_dim)
    def forward(self, x, skip, t_emb=None, context=None):
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, context)
        x = self.res2(x, t_emb)
        return x

class UNet3DConditionModel_maison(nn.Module):
    def __init__(
        self,
        sample_size: Tuple[int, int, int],
        in_channels: int = 1,
        out_channels: int = 1,
        down_block_types: Sequence[str] = ("DownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
        up_block_types: Sequence[str] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "UpBlock3D"),
        block_out_channels: Sequence[int] = (32, 64, 128, 256),
        cross_attention_dim: int = 512,
        attention_head_dim: int = 64,
        time_embedding_dim: int = 512,
    ):
        super().__init__()
        assert len(down_block_types) == len(up_block_types) == len(block_out_channels), \
            "down_block_types, up_block_types and block_out_channels must have same length"
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.cross_attention_dim = cross_attention_dim
        self.attention_head_dim = attention_head_dim
        self.time_embedding_dim = time_embedding_dim

        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        prev_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            block_type = down_block_types[i]
            use_attn = "CrossAttn" in block_type or "Attn" in block_type
            num_heads = max(1, out_ch // attention_head_dim)
            head_dim = attention_head_dim
            db = DownBlock3D(prev_ch, out_ch, time_emb_dim=time_embedding_dim,
                             use_attn=use_attn, num_heads=num_heads, head_dim=head_dim,
                             cross_attention_dim=cross_attention_dim if use_attn else None)
            self.down_blocks.append(db)
            if i != len(block_out_channels) - 1:
                self.downsamplers.append(Downsample3D(out_ch))
            prev_ch = out_ch

        mid_ch = block_out_channels[-1]
        self.mid_block1 = ResidualBlock3D(mid_ch, mid_ch, time_emb_dim=time_embedding_dim)
        self.mid_attn = AttentionBlock3D(mid_ch, num_heads=max(1, mid_ch // attention_head_dim),
                                         head_dim=attention_head_dim, cross_attention_dim=cross_attention_dim)
        self.mid_block2 = ResidualBlock3D(mid_ch, mid_ch, time_emb_dim=time_embedding_dim)

        self.upsamplers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        rev_out = list(reversed(block_out_channels))
        prev_ch = rev_out[0]
        for i, out_ch in enumerate(rev_out):
            block_type = up_block_types[i]
            use_attn = "CrossAttn" in block_type or "Attn" in block_type
            num_heads = max(1, out_ch // attention_head_dim)
            head_dim = attention_head_dim
            in_ch = prev_ch + out_ch
            ub = UpBlock3D(in_ch, out_ch, time_emb_dim=time_embedding_dim,
                           use_attn=use_attn, num_heads=num_heads, head_dim=head_dim,
                           cross_attention_dim=cross_attention_dim if use_attn else None)
            self.up_blocks.append(ub)
            if i != len(rev_out) - 1:
                self.upsamplers.append(Upsample3D(out_ch))
            prev_ch = out_ch

        self.norm_out = nn.GroupNorm(8, block_out_channels[0])
        self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None):
        """
        sample: (B, C, D, H, W)
        timestep: (B,) or scalar
        encoder_hidden_states: (B, M, cross_attention_dim)
        """
        if timestep.dim() == 0:
            timestep = timestep.view(1).expand(sample.shape[0])
        t_emb = timestep_embedding(timestep, self.time_embedding_dim, max_period=10000)
        t_emb = self.time_mlp(t_emb)

        x = self.conv_in(sample)
        skips = []

        for i, db in enumerate(self.down_blocks):
            x = db(x, t_emb, encoder_hidden_states)
            skips.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, encoder_hidden_states)
        x = self.mid_block2(x, t_emb)

        for i, ub in enumerate(self.up_blocks):
            skip = skips.pop()
            x = ub(x, skip, t_emb, encoder_hidden_states)
            if i < len(self.upsamplers):
                x = self.upsamplers[i](x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

# ----------------------
# Inference script
# ----------------------

parser = argparse.ArgumentParser(description="Run diffusion script with custom lambda guidance.")
parser.add_argument("--lamdba", type=float, default=0.6, help="Guidance lambda value (default: 0.8)")
parser.add_argument("--brain_folder", type=str, default="/path/to/input", help="Folder with volumes to harmonize")
parser.add_argument("--save_dir", type=str, default="/path/to/output", help="Folder to save harmonized volumes")
args = parser.parse_args()

lamdba = args.lamdba
brain_folder = args.brain_folder
save_dir = args.save_dir
print(lamdba)
print(brain_folder)
print(save_dir)

# checkpoint (replaced local path with generic path)
checkpoint_path = "/path/to/checkpoint.pt"
checkpoint = torch.load(checkpoint_path)

# diffusion model setup
patch_size = (80, 96, 80)

model_diffusion = UNet3DConditionModel_maison(
    sample_size=patch_size,
    in_channels=3,
    out_channels=1,
    down_block_types=("DownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
    up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "UpBlock3D"),
    block_out_channels=(64, 128, 256, 256),
    cross_attention_dim=512,
    attention_head_dim=64,
    time_embedding_dim=512,
)

model_diffusion.load_state_dict(checkpoint['model_state_dict'])

# embedder
n_classes = len(ds2id)
embedder = nn.Embedding(n_classes + 1, 512)
embedder.load_state_dict(checkpoint['embedder_state_dict'])

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

img_list = os.listdir(brain_folder)
path_img_list = [f"{brain_folder}/{img_name}" for img_name in img_list]

for path_img in tqdm(path_img_list):
    img_nib = nib.load(path_img)
    img = img_nib.get_fdata()
    img_name = os.path.basename(path_img).split('.')[0]

    v_mean = img.mean()
    v_std = img.std()
    img = (img - v_mean) / v_std

    volume = torch.tensor(img, device=accelerator.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    label_cat = ds2id['aibl_gamma_0']
    volume_anat_map = make_structural_anatomy_map(volume, grad_sigmas=(0.5, 1.0),
                                                 hf_sigma=0.7, smooth_sigma=0.0, normalize_percentiles=(0.5, 99.5))
    anat_coarse = F.interpolate(volume_anat_map.float(), size=patch_size, mode='nearest')

    pd, ph, pw = patch_size
    device = accelerator.device

    model_diffusion, embedder, volume, volume_anat_map = accelerator.prepare(
        model_diffusion, embedder, volume, volume_anat_map
    )
    model_diffusion.eval()

    # tight crop to non-zero region
    zmin, zmax, ymin, ymax, xmin, xmax = compute_tight_crop_coords(volume)
    crop = volume[:, :, zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    crop_anat = volume_anat_map[:, :, zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
    mask_brain = ~torch.isclose(crop, crop.min(), atol=1e-6)
    mask_brain = mask_brain.float()

    _, _, Dc, Hc, Wc = crop.shape
    zs_pos = compute_positions_no_pad(Dc, pd)
    ys_pos = compute_positions_no_pad(Hc, ph)
    xs_pos = compute_positions_no_pad(Wc, pw)

    w_full = make_blending_window(patch_size).to(device)

    output_crop = torch.zeros_like(crop, dtype=torch.float32, device=device)
    weight_crop = torch.zeros_like(crop, dtype=torch.float32, device=device)

    all_patch_mins = []
    all_patch_maxs = []

    print(path_img)

    all_generated_patches = []
    all_patch_coords = []

    label_ids_eval = torch.tensor([label_cat], device=device, dtype=torch.long)
    cond_label_embedding = embedder(label_ids_eval).unsqueeze(1)
    uncond_ids = torch.zeros_like(label_ids_eval, dtype=torch.long)
    uncond_label_embedding = embedder(uncond_ids).unsqueeze(1)

    num_inference_steps = 50
    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps_iter = list(noise_scheduler.timesteps)
    num_train_timesteps = noise_scheduler.config.num_train_timesteps
    step_offset = num_train_timesteps // num_inference_steps
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    final_alpha_cumprod = noise_scheduler.final_alpha_cumprod.to(device)

    for z in zs_pos:
        for y in ys_pos:
            for x in xs_pos:
                # extract patch (no padding)
                patch_vol = crop[:, :, z:z + pd, y:y + ph, x:x + pw]
                patch_anat = crop_anat[:, :, z:z + pd, y:y + ph, x:x + pw]

                gaussian_noise = torch.randn_like(patch_vol).to(device)
                batch_size = 1

                with torch.no_grad():
                    with accelerator.autocast():
                        for t in timesteps_iter:
                            t_b = torch.tensor([int(t)] * batch_size, device=device)

                            volume_in = torch.cat([gaussian_noise, gaussian_noise], dim=0)
                            anatomy_in = torch.cat([patch_anat, patch_anat], dim=0)
                            coarse_in = torch.cat([patch_anat, patch_anat], dim=0)

                            model_input_eval = torch.cat([volume_in, anatomy_in, coarse_in], dim=1)
                            emb_in = torch.cat([uncond_label_embedding, cond_label_embedding], dim=0)
                            t_in = torch.cat([t_b, t_b], dim=0)

                            model_output_eval = model_diffusion(model_input_eval, t_in, encoder_hidden_states=emb_in)
                            uncond_noise_pred, cond_noise_pred = model_output_eval.chunk(2, dim=0)

                            prev_t = t - step_offset
                            alpha_t = alphas_cumprod[t]
                            alpha_prev = alphas_cumprod[prev_t] if prev_t >= 0 else final_alpha_cumprod
                            beta_t = 1 - alpha_t

                            pred = uncond_noise_pred + lamdba * (cond_noise_pred - uncond_noise_pred)
                            x0_pred = (gaussian_noise - beta_t**0.5 * pred) / alpha_t**0.5

                            coeff_dir = (1 - alpha_prev)**0.5
                            pred_sample_direction = coeff_dir * uncond_noise_pred

                            gaussian_noise = alpha_prev**0.5 * x0_pred + pred_sample_direction

                diffused_patch = gaussian_noise.detach().float().cpu()
                all_generated_patches.append(diffused_patch)
                all_patch_coords.append((z, y, x))

    # reconstruction and harmonization of patches
    eps = 1e-6

    # We use the brain mask to ensure homogenous background
    # erosion: apply mean kernel and strict threshold to reduce mask
    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    mask_eroded = F.conv3d(mask_brain, kernel, padding=1)
    mask_eroded = (mask_eroded >= 27.0).float()

    # smooth eroded contours
    kernel_smooth = torch.ones((1, 1, 3, 3, 3), device=device) / 27.0
    mask_brain_smooth = F.conv3d(mask_eroded, kernel_smooth, padding=1).clamp(0.0, 1.0)

    # global stats on brain voxels
    brain_mask_bool = mask_brain_smooth.bool()
    brain_voxels = crop[brain_mask_bool]

    global_mean = brain_voxels.mean()
    global_std = brain_voxels.std().clamp(min=eps)

    w = w_full.view(1, 1, pd, ph, pw)

    output_crop.zero_()
    weight_crop.zero_()
    all_patch_mins.clear()
    all_patch_maxs.clear()

    for idx, patch_cpu in enumerate(all_generated_patches):
        z, y, x = all_patch_coords[idx]

        patch = patch_cpu.to(device).float()
        if patch.dim() == 4:
            patch = patch.unsqueeze(0)

        patch_mask = mask_brain_smooth[:, :, z:z + pd, y:y + ph, x:x + pw]
        patch_mask_bool = patch_mask.bool()

        if patch_mask_bool.any():
            vox = patch[patch_mask_bool]
        else:
            vox = patch.reshape(-1)

        p_mean = vox.mean()
        p_std = vox.std().clamp(min=eps)

        # match patch stats to global stats
        patch_matched = (patch - p_mean) / p_std * global_std + global_mean
        patch_matched = torch.clamp(patch_matched, min=crop.min().item(), max=crop.max().item())

        all_patch_mins.append(float(patch_matched.min().cpu().item()))
        all_patch_maxs.append(float(patch_matched.max().cpu().item()))

        weighted_patch = patch_matched * w
        weight_mask = (w * patch_mask)

        output_crop[:, :, z:z + pd, y:y + ph, x:x + pw] += weighted_patch * patch_mask
        weight_crop[:, :, z:z + pd, y:y + ph, x:x + pw] += weight_mask

    # safe division by weights
    small = 1e-8
    safe_weight = weight_crop.clone()
    safe_weight[safe_weight < small] = 1.0

    reconstructed = output_crop / safe_weight
    reconstructed = reconstructed * mask_brain_smooth

    # place reconstruction into full volume
    full_volume = torch.zeros((1, 1, 160, 192, 160), device=reconstructed.device, dtype=reconstructed.dtype)
    full_volume[:, :, zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1] = reconstructed

    # normalize brain voxels only (keep background zero)
    full_mask = torch.zeros_like(full_volume, dtype=torch.bool)
    full_mask[:, :, zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1] = mask_brain_smooth.bool()

    brain_vals = full_volume[full_mask]
    if brain_vals.numel() > 0:
        vmin = brain_vals.min()
        vmax = brain_vals.max()
        denom = (vmax - vmin).clamp(min=eps)
        full_volume_norm = torch.zeros_like(full_volume)
        full_volume_norm[full_mask] = (full_volume[full_mask] - vmin) / denom
    else:
        full_volume_norm = full_volume

    full_volume = full_volume_norm
    full_volume = full_volume.cpu().detach().squeeze().numpy()

    recon_nib = nib.Nifti1Image(full_volume, affine=img_nib.affine, header=img_nib.header)
    nib.save(recon_nib, f"{save_dir}/{img_name}.nii.gz")
