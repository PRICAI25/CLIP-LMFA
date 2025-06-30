import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=226):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),  # Ensure c_in matches the input dimension
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),  # Ensure c_in matches the output dimension
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        # Ensure x has the correct shape for the linear layer
        # print("Input shape:", x.shape)  # Debugging line
        x = x.squeeze(1)  # Remove the singleton dimension
        # print("Input shape——two:", x.shape)  # Debugging line
        x = self.fc1(x)
        # print("Output shape:", x.shape)
        y = self.fc2(x)
        # print("Output shape——two:", y.shape)
        return x, y

class CLIP_Inplanted(nn.Module):
    def __init__(self, gem_model, features, device):
        super().__init__()
        self.gem_model = gem_model
        self.image_encoder = gem_model.model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList([ClipAdapter(896, bottleneck=226) for i in range(len(features))])
        self.det_adapters = nn.ModuleList([ClipAdapter(896, bottleneck=226) for i in range(len(features))])
        self.device = device
        # Move the model to the specified device
        self.to(device)

    def forward(self, x):
        # Ensure x is on the same device as the model
        x = x.to(self.device)
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        attn_out = []
        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            if i + 1 == 12:
                print("image_encoder: ", self.image_encoder.transformer.resblocks[i](x, attn_mask=None))
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                attn_out.append(attn)
            else:
                # Directly assign the returned tensor to x
                print("image_encoder_two: ", self.image_encoder.transformer.resblocks[i](x, attn_mask=None))
                x = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)  # No need to unpack
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)

        B, C, L = attn_out[0].shape
        H = int(math.sqrt(L-1))
        out_attn = torch.zeros([H, H]).to(self.device)  # Ensure out_attn is on the same device as x

        for i in range(len(attn)):
            out_attn = out_attn + attn_out[i][0, 0, 1:].view(H, H)
        x = x.permute(1, 0, 2)

        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        # Normalize the outputs
        features_gem = F.normalize(pooled, dim=-1)
        features_clip = F.normalize(tokens, dim=-1)

        return features_gem, features_clip

