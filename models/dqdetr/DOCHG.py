"""
Density-Guided Object Center Heatmap Generator (DOCHG)

Combines:
  - ObjectCenterFeatureGenerator (formerly BinarySupervision): generates object center heatmap features
  - get_dochg_loss (formerly get_DDG_Loss): density-guided Gaussian heatmap loss
  - get_density_weights (formerly from DGL): density-based weighting for each bbox
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================================
# Helper functions for density-guided Gaussian heatmap loss
# ============================================================================

mse_loss = torch.nn.MSELoss()


def process_map(map, size):
    map = F.interpolate(map, size=size, mode='nearest')

    map = torch.mean(map, dim=0)
    map = torch.mean(map, dim=0)
    map = (map - map.min()) / (map.max() - map.min())

    return map


def generate_density_gauss_map(target, density_map, device):
    image_size = target['size'].cpu()
    height, width = image_size

    density_gauss_map = np.zeros((height, width), dtype=np.float32)

    bboxes = target['boxes'].tolist()
    labels = target['labels'].tolist()
    filtered = [(box, label) for box, label in zip(bboxes, labels) if label != 0]

    bboxes, labels = zip(*filtered) if filtered else ([], [])

    areas = [w * h for x, y, w, h in bboxes]
    max_area = max(areas) if areas else 1.0

    for bbox in bboxes:
        x, y, w, h = bbox

        x_center = int(x * width)
        y_center = int(y * height)
        w_bbox = int(w * width)
        h_bbox = int(h * height)

        x_start = max(0, int(x_center - w_bbox / 2))
        x_end = min(width, int(x_center + w_bbox / 2))
        y_start = max(0, int(y_center - h_bbox / 2))
        y_end = min(height, int(y_center + h_bbox / 2))

        if x_start >= x_end or y_start >= y_end:
            continue

        density_patch = density_map[y_start:y_end, x_start:x_end]
        local_density = torch.mean(density_patch).item()

        b = 4.0
        base_x = w_bbox / b
        base_y = h_bbox / b
        density_scale = 1.0

        sigma_x = base_x / (1 + density_scale * local_density)
        sigma_y = base_y / (1 + density_scale * local_density)

        y_indices = np.arange(y_start, y_end)
        x_indices = np.arange(x_start, x_end)
        yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')

        dx = xx - x_center
        dy = yy - y_center

        exponent = (dx**2) / (2 * sigma_x**2) + (dy**2) / (2 * sigma_y**2)
        gaussian = np.exp(-exponent)

        density_gauss_map[y_start:y_end, x_start:x_end] = np.maximum(
            density_gauss_map[y_start:y_end, x_start:x_end],
            gaussian * math.log(max_area / (h * w))
        )

    target['density_gauss_map'] = torch.tensor(density_gauss_map, dtype=torch.float32, device=device)

    return target


def get_dochg_loss(oc_map, density_map, targets, alpha=1):
    density_map = process_map(density_map, tuple(targets[0]['size']))
    oc_map = process_map(oc_map, tuple(targets[0]['size']))
    generate_density_gauss_map(targets[0], density_map, oc_map.device)

    dochg_loss = mse_loss(oc_map, targets[0]['density_gauss_map'])

    return dochg_loss * alpha


# ============================================================================
# Density weights for bbox-level weighting
# ============================================================================

def get_density_weights(density_map: torch.Tensor, bboxes: torch.Tensor):
    B, C, H, W = density_map.shape
    _, Q, _ = bboxes.shape

    avg_density = density_map.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    avg_density_flat = avg_density.view(B, -1)
    min_val = avg_density_flat.min(dim=1)[0].view(B, 1, 1, 1)
    max_val = avg_density_flat.max(dim=1)[0].view(B, 1, 1, 1)
    norm_density = (avg_density - min_val) / (max_val - min_val + 1e-6)  # [B, 1, H, W]

    cx = (bboxes[:, :, 0] * W).long().clamp(0, W - 1)  # [B, Q]
    cy = (bboxes[:, :, 1] * H).long().clamp(0, H - 1)  # [B, Q]

    weights = []
    for b in range(B):
        b_weights = []
        for q in range(Q):
            cx, cy, w, h = bboxes[b, q]
            x1 = int(((cx - w / 2) * W).clamp(0, W - 1).item())
            y1 = int(((cy - h / 2) * H).clamp(0, H - 1).item())
            x2 = int(((cx + w / 2) * W).clamp(1, W).item())
            y2 = int(((cy + h / 2) * H).clamp(1, H).item())

            region = norm_density[b, 0, y1:y2, x1:x2]
            if region.numel() == 0:
                mean_val = torch.tensor(0.0, device=density_map.device)
            else:
                mean_val = region.mean()

            b_weights.append(mean_val + 1)
        weights.append(torch.stack(b_weights))  # [Q]
    weights = torch.stack(weights, dim=0).unsqueeze(-1)  # [B, Q, 1]

    return weights


# ============================================================================
# Object Center Feature Generator (formerly BinarySupervision)
# ============================================================================

def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


class ObjectCenterFeatureGenerator(nn.Module):
    def __init__(self):
        super(ObjectCenterFeatureGenerator, self).__init__()
        self.ccm_cfg = [512, 512, 512, 256, 256, 256]
        self.in_channels = 512
        self.conv1 = nn.Conv2d(256, self.in_channels, kernel_size=1)
        self.ccm = make_layers(self.ccm_cfg, in_channels=self.in_channels, d_rate=2)

    def forward(self, features, spatial_shapes=None):
        features = features.transpose(1, 2)
        bs, c, hw = features.shape
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]

        v_feat = features[:, :, 0:h * w].view(bs, 256, h, w)
        x = self.conv1(v_feat)
        x = self.ccm(x)

        return x

