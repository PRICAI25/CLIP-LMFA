"""
  @Author: 王权
  @FileName: SCSA.py
  @DateTime: 2024/12/12 18:10
  @SoftWare: PyCharm
"""
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math




# 通道注意力模块
# 通道注意力模块
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         # 确保中间层的通道数至少为1
#         mid_channels = max(1, in_planes // ratio)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 共享MLP
#         self.fc1 = nn.Conv2d(in_planes, mid_channels, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(mid_channels, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
#
#
# # 空间注意力模块
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
# # CBAM模块
# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_planes, ratio)
#         self.spatial_attention = SpatialAttention(kernel_size)
#         self.weighted_residual_channel = WeightedResidualConnection(in_planes)
#         self.weighted_residual_spatial = WeightedResidualConnection(in_planes)
#
#     def forward(self, x):
#         # 保存原始输入用于残差连接
#         identity = x
#
#         # 应用通道注意力并进行加权残差连接
#         x = self.channel_attention(x)
#         x = self.weighted_residual_channel(x, identity)
#
#         # 应用空间注意力并进行加权残差连接
#         x = self.spatial_attention(x)
#         x = self.weighted_residual_spatial(x, identity)
#
#         return x
#
# class WeightedResidualConnection(nn.Module):
#     def __init__(self, in_channels):
#         super(WeightedResidualConnection, self).__init__()
#         # 初始化可学习的权重参数
#         self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1))
#
#     def forward(self, x, identity):
#         # 使用可学习的权重参数进行加权残差连接
#         return self.alpha * x + identity



# -------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SimpleChannelAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.SiLU(),
            nn.Linear(channel, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.avg_pool(x).view(b, c)
        attention = self.fc(avg_out).view(b, c, 1, 1)
        return x * attention


class SimpleSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


class SimpleCBAM(nn.Module):
    def __init__(self, in_channels=3, feature_dim=None, seq_length=None):
        super().__init__()
        self.ca = SimpleChannelAttention(in_channels)
        self.sa = SimpleSpatialAttention()
        self.feature_dim = feature_dim
        self.seq_length = seq_length

        if feature_dim and seq_length:
            self.proj = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),  # 简化池化大小
                nn.Flatten(),
                nn.Linear(in_channels * 16, feature_dim),
                nn.SiLU()
            )

    def forward(self, x):
        # 通道注意力
        out = self.ca(x)
        # 空间注意力
        out = self.sa(out)

        if self.feature_dim and self.seq_length:
            # 投影到目标维度
            features = self.proj(out)
            # 扩展到序列长度
            features = features.unsqueeze(1).expand(-1, self.seq_length, -1)
            return features, features  # 返回相同的特征用于gem和clip

        return out


import torch
import torch.nn as nn
import math


class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


# Channel and Position Attention Mechanism (CPAM)
class CPAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, images, alpha=0.5, features_gem=None, features_clip=None):
        # Apply channel attention
        enhanced_gem = self.channel_att(images)

        # Combine with features_gem
        if features_gem is not None:
            features_gem = alpha * features_gem + (1 - alpha) * enhanced_gem

        # Combine with features_clip
        if features_clip is not None:
            enhanced_clip = self.local_att(enhanced_gem)  # Assuming you want to apply local attention to enhanced_gem
            features_clip = alpha * features_clip + (1 - alpha) * enhanced_clip

        return features_gem, features_clip