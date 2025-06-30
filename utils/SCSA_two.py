"""
  @Author: 王权
  @FileName: SCSA_two.py
  @DateTime: 2024/12/20 13:03
  @SoftWare: PyCharm
"""

import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import init

import torch
import torch.nn as nn

# class MultiScaleModule(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#
#         # Increase the number of output channels for better feature extraction
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels * 20, kernel_size=1),
#             nn.BatchNorm2d(in_channels * 20),
#             nn.SiLU()
#         )
#
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels * 20, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels * 20),
#             nn.SiLU()
#         )
#
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels * 20, kernel_size=5, padding=2),
#             nn.BatchNorm2d(in_channels * 20),
#             nn.SiLU()
#         )
#
#         # Feature fusion with increased channels
#         self.fusion = nn.Sequential(
#             nn.Conv2d(in_channels * 60, in_channels, kernel_size=1),  # Adjusted for increased channels
#             nn.BatchNorm2d(in_channels),
#             nn.SiLU()
#         )
#
#         # Optional: Add dropout for regularization
#         self.dropout = nn.Dropout(p=0.3)
#
#     def forward(self, x):
#         b1 = self.branch1(x)
#         b2 = self.branch2(x)
#         b3 = self.branch3(x)
#
#         # Concatenate features from all branches
#         multi_scale_features = torch.cat([b1, b2, b3], dim=1)
#         output = self.fusion(multi_scale_features)
#
#         # Apply dropout
#         output = self.dropout(output)
#
#         return output + x  # Residual connection
import torch
import torch.nn as nn

class MultiScaleModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Reduce the number of output channels for efficiency
        out_channels = in_channels * 4  # Adjusted for efficiency

        # Use depthwise separable convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        # Feature fusion with reduced channels
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        # Optional: Add dropout for regularization
        self.dropout = nn.Dropout(p=0.1)

        # Optional: Add SE block for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # Concatenate features from all branches
        multi_scale_features = torch.cat([b1, b2, b3], dim=1)
        output = self.fusion(multi_scale_features)

        # Apply dropout
        output = self.dropout(output)

        # Apply SE block
        se_weight = self.se(output)
        output = output * se_weight

        return output + x  # Residual connection
import torch
import torch.nn as nn
#
# class MultiScaleModule(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#
#         # Reduce the number of output channels for efficiency
#         out_channels = in_channels * 4
#
#         # Use depthwise separable convolutions
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU()
#         )
#
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
#             nn.Conv2d(in_channels , out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU()
#         )
#
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU()
#         )
#
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU()
#         )
#
#         # Feature fusion with reduced channels
#         self.fusion = nn.Sequential(
#             nn.Conv2d(out_channels * 4, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.SiLU()
#         )
#
#         # Optional: Add dropout for regularization
#         self.dropout = nn.Dropout(p=0.1)
#
#         # Optional: Add SE block for channel attention
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
#             nn.SiLU(),
#             nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b1 = self.branch1(x)
#         b2 = self.branch2(x)  # Residual connection in branch
#         b3 = self.branch3(x)   # Residual connection in branch
#         b4 = self.branch4(x)   # Residual connection in branch
#
#         # Concatenate features from all branches
#         multi_scale_features = torch.cat([b1, b2, b3, b4], dim=1)
#         output = self.fusion(multi_scale_features)
#
#         # Apply dropout
#         output = self.dropout(output)
#
#         # Apply SE block
#         se_weight = self.se(output)
#         output = output * se_weight
#
#         return output + x  # Residual connection




# class SimpleChannelAttention(nn.Module):
#     def __init__(self, channel):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // 16, bias=False),
#             nn.SiLU(),
#             nn.Linear(channel // 16, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.shape
#         avg_out = self.avg_pool(x).view(b, c)
#         attention = self.fc(avg_out).view(b, c, 1, 1)
#         return x * attention




class ChannelAttentionBlock(nn.Module):
    # 实验不同的超参数：尝试不同的 reduction 值，以找到最佳的通道压缩比例。
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttentionBlock, self).__init__()

        # 左侧分支
        self.left_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),  # 1x1 卷积
            nn.ReLU(inplace=True),  # ReLU 激活
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)  # 1x1 卷积
        )

        # 右侧分支
        self.right_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # 自适应平均池化
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),  # 1x1 卷积
            nn.ReLU(inplace=True),  # ReLU 激活
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)  # 1x1 卷积
        )

        # Sigmoid 激活
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x.shape:", x.shape)
        # 左侧分支输出
        left_out = self.left_branch(x)
        # print("left_out.shape:", left_out.shape)
        # 右侧分支输出
        right_out = self.right_branch(x)
        # print("right_out.shape:", right_out.shape)
        # 融合与激活
        out = left_out + right_out  # 加法操作
        # print( "out.shape:", out.shape)
        attention = self.sigmoid(out)  # Sigmoid 激活
        # print( "attention.shape:", attention.shape)
        # 输出
        # print("x * attention.shape:", x * attention.shape)
        return x * attention  # 通道注意力权重应用


# x.shape: torch.Size([1, 64, 240, 240])
# avg_out.shape: torch.Size([1, 1, 240, 240])
# max_out.shape: torch.Size([1, 1, 240, 240])
# x.shape: torch.Size([1, 2, 240, 240])
# xconv.shape: torch.Size([1, 16, 240, 240])
# attention.shape: torch.Size([1, 1, 240, 240])
# x.shape: torch.Size([1, 64, 240, 240])
# left_out.shape: torch.Size([1, 64, 1, 1])
# right_out.shape: torch.Size([1, 64, 1, 1])
# out.shape: torch.Size([1, 64, 1, 1])
# attention.shape: torch.Size([1, 64, 1, 1])
# x.shape: torch.Size([1, 64, 448, 448])
# avg_out.shape: torch.Size([1, 1, 448, 448])
# max_out.shape: torch.Size([1, 1, 448, 448])
# x.shape: torch.Size([1, 2, 448, 448])
# xconv.shape: torch.Size([1, 16, 448, 448])
# attention.shape: torch.Size([1, 1, 448, 448])
# x.shape: torch.Size([1, 64, 448, 448])
# left_out.shape: torch.Size([1, 64, 1, 1])
# right_out.shape: torch.Size([1, 64, 1, 1])
# out.shape: torch.Size([1, 64, 1, 1])
# attention.shape: torch.Size([1, 64, 1, 1])
# x.shape: torch.Size([1, 64, 896, 896])
# avg_out.shape: torch.Size([1, 1, 896, 896])
# max_out.shape: torch.Size([1, 1, 896, 896])
# x.shape: torch.Size([1, 2, 896, 896])
# xconv.shape: torch.Size([1, 16, 896, 896])
# attention.shape: torch.Size([1, 1, 896, 896])
# x.shape: torch.Size([1, 64, 896, 896])
# left_out.shape: torch.Size([1, 64, 1, 1])
# right_out.shape: torch.Size([1, 64, 1, 1])
# out.shape: torch.Size([1, 64, 1, 1])
# attention.shape: torch.Size([1, 64, 1, 1])


class SimpleSpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)  # 添加 Batch Normalization
            self.relu = nn.ReLU(inplace=True)  # 使用 ReLU 激活
            self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(1)  # 添加 Batch Normalization
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x_1 = x
            # print("x.shape:", x.shape)
            avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算平均池化
            # print("avg_out.shape:", avg_out.shape)
            max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算最大池化
            # print("max_out.shape:", max_out.shape)
            x = torch.cat([avg_out, max_out], dim=1)  # 拼接平均和最大池化结果
            # print("x.shape:", x.shape)
            x = self.conv1(x)  # 第一个卷积层
            # print("xconv.shape:", x.shape)
            x = self.bn1(x)  # Batch Normalization
            x = self.relu(x)  # 激活函数

            x = self.conv2(x)  # 第二个卷积层
            x = self.bn2(x)  # Batch Normalization
            attention = self.sigmoid(x)  # Sigmoid 激活
            # print("attention.shape:", attention.shape)
            return x_1 * attention  # 应用注意力权重


import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GatedAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_gate = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算注意力特征
        attention = self.conv1(x)
        gate = self.conv_gate(x)
        gate = self.sigmoid(gate)  # 使用 Sigmoid 激活函数

        # 应用门控机制
        return x * attention * gate  # 特征与注意力和门控相乘

import torch
import torch.nn as nn

class ImprovedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU()  # 使用 SiLU 激活函数
        self.dropout = nn.Dropout(0.1)  # 添加 Dropout 以防止过拟合

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

# 自注意力
# class ContextAwareAdapter(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.context_layer = nn.Sequential(
#             nn.Linear(input_dim, input_dim // 2),
#             nn.ReLU(),
#             nn.Linear(input_dim // 2, input_dim)
#         )
#         # 添加一个自注意力层
#         self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
#
#         # 计算全局上下文信息
#         context_info = x.mean(dim=[2, 3])  # 全局平均池化
#
#         # 应用自注意力机制
#         context_info = context_info.unsqueeze(0).permute(1, 0, 2)  # 调整维度以适应自注意力机制
#         attn_output, _ = self.self_attention(context_info, context_info, context_info)
#         context_info = attn_output.permute(1, 0, 2).squeeze(1)  # 恢复原维度
#
#         # 通过上下文层
#         context_enhanced = self.context_layer(context_info)
#
#         # 将上下文信息扩展回原始特征图的大小
#         context_enhanced = context_enhanced.view(batch_size, channels, 1, 1)
#         return x + context_enhanced


# class ContextAwareAdapter(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.context_layer = nn.Sequential(
#             nn.Linear(input_dim, input_dim // 2),
#             nn.ReLU(),
#             nn.Linear(input_dim // 2, input_dim),
#             nn.Sigmoid()  # 添加sigmoid激活函数以生成权重
#         )
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
#
#         # 计算全局上下文信息
#         context_info = x.mean(dim=[2, 3])
#
#         # 通过上下文层
#         context_weights = self.context_layer(context_info)
#
#         # 将上下文权重扩展回原始特征图的大小
#         context_weights = context_weights.view(batch_size, channels, 1, 1)
#
#         # 使用乘法来融合上下文信息
#         return x * context_weights

# 门控
class ContextAwareAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.context_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # 计算全局上下文信息
        context_info = x.mean(dim=[2, 3])

        # 通过上下文层
        context_weights = self.context_layer(context_info)

        # 计算门控值
        gate_value = self.gate(context_info).view(batch_size, channels, 1, 1)

        # 使用门控机制来融合上下文信息
        return gate_value * x + (1 - gate_value) * context_weights.view(batch_size, channels, 1, 1)
# 普通
# class ContextAwareAdapter(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.context_layer = nn.Sequential(
#             nn.Linear(input_dim, input_dim // 2),
#             nn.ReLU(),
#             nn.Linear(input_dim // 2, input_dim)
#         )
#
#     def forward(self, x):
#         # 打印输入形状
#         # print(f"Input shape: {x.shape}")
#
#         # 计算全局上下文信息，输出形状为 (batch_size, channels)
#         context_info = x.mean(dim=[2, 3])  # 计算全局上下文信息，输出形状为 (batch_size, channels)
#
#         # 打印上下文信息的形状
#         # print(f"Context info shape: {context_info.shape}")
#
#         # 通过上下文层
#         context_enhanced = self.context_layer(context_info)
#         # print(f"Context enhanced shape: {context_enhanced.shape}")
#
#         # 将上下文信息扩展到 (batch_size, channels, 1, 1)
#         return x + context_enhanced.view(context_enhanced.size(0), -1, 1, 1)
import torch
import torch.nn as nn

class LocalFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(LocalFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 提取局部特征
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        return x
# 主要
# 总结来说，这个类实现了一种根据输入数据类别动态调整处理方式的方法，通过多个并行的适配器处理输入，并依据预测的类别概率将这些处理结果融合起来
class CategorySpecificAdapter(nn.Module):
    def __init__(self, input_dim, num_categories):
        super().__init__()
        # 创建一个 ModuleList 对象 adapters，里面包含 num_categories 个 ContextAwareAdapter 实例。每个实例都是基于 input_dim 初始化的，这意味着为每一个类别创建了一个独立的适配器
        self.adapters = nn.ModuleList([ContextAwareAdapter(input_dim) for _ in range(num_categories)])
        # 定义了一个序列模型 category_classifier，包括一个线性层（将输入维度映射到类别数量）和一个 Softmax 层，用于计算属于各个类别的概率。
        self.category_classifier = nn.Sequential(
            nn.Linear(input_dim, num_categories),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 获取类别概率
        batch_size, channels, height, width = x.shape
        # 对输入 x 进行平均池化操作（在高度和宽度维度上），然后通过 category_classifier 来获取每个样本属于各个类别的概率
        category_probs = self.category_classifier(x.mean(dim=[2, 3]))

        # 应用类别特定适配器
        adapted_features = []
        # 遍历所有适配器，并让每个适配器独立地处理输入 x。这里的关键是所有的适配器并行处理输入 x，而不是串行处理。
        for i, adapter in enumerate(self.adapters):
            adapted = adapter(x)
            adapted_features.append(adapted)

        # 根据类别概率融合特征   创建一个与 adapted_features[0] 形状相同的零张量 fused_feature
        fused_feature = torch.zeros_like(adapted_features[0])
        # 根据前面计算出的类别概率，以加权求和的方式将所有适配器的输出进行融合。权重就是对应类别的概率值，通过广播机制将概率应用于相应的特征图。
        for i in range(len(self.adapters)):
            fused_feature += category_probs[:, i].view(batch_size, 1, 1, 1) * adapted_features[i]

        return fused_feature

class CategorySpecificAdapter_multi(nn.Module):
    def __init__(self, input_dim, num_categories):
        super().__init__()
        self.adapters = nn.ModuleList([MultiScaleModule(input_dim) for _ in range(num_categories)])
        self.category_classifier = nn.Sequential(
            nn.Linear(input_dim, num_categories),
            nn.Dropout(0.5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 获取类别概率
        batch_size, channels, height, width = x.shape
        category_probs = self.category_classifier(x.mean(dim=[2, 3]))

        # 应用类别特定适配器
        adapted_features = []
        for i, adapter in enumerate(self.adapters):
            adapted = adapter(x)
            adapted_features.append(adapted)

        # 根据类别概率融合特征
        fused_feature = torch.zeros_like(adapted_features[0])
        for i in range(len(self.adapters)):
            fused_feature += category_probs[:, i].view(batch_size, 1, 1, 1) * adapted_features[i]

        return fused_feature
import torch
import torch.nn as nn
import torch.nn.functional as F
# CBAM的分类器
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.avg_pool(x)
#         avg_out = self.fc1(avg_out)
#         avg_out = self.relu(avg_out)
#         avg_out = self.fc2(avg_out)
#
#         max_out = self.max_pool(x)
#         max_out = self.fc1(max_out)
#         max_out = self.relu(max_out)
#         max_out = self.fc2(max_out)
#
#         out = avg_out + max_out
#         out = self.sigmoid(out)
#         return out
#
# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv1(out)
#         out = self.sigmoid(out)
#         return out
#
# class CBAM(nn.Module):
#     def __init__(self, in_planes):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_planes)
#         self.spatial_attention = SpatialAttention()
#
#     def forward(self, x):
#         x = self.channel_attention(x) * x
#         x = self.spatial_attention(x) * x
#         return x
#
# class CategorySpecificAdapter(nn.Module):
#     def __init__(self, input_dim, num_categories):
#         super().__init__()
#         self.adapters = nn.ModuleList([CBAM(input_dim) for _ in range(num_categories)])
#         self.category_classifier = nn.Sequential(
#             nn.Linear(input_dim, num_categories),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         # 获取类别概率
#         batch_size, channels, height, width = x.shape
#         category_probs = self.category_classifier(x.mean(dim=[2, 3]))
#
#         # 应用类别特定适配器
#         adapted_features = []
#         for adapter in self.adapters:
#             adapted = adapter(x)
#             adapted_features.append(adapted)
#
#         # 根据类别概率融合特征
#         fused_feature = torch.zeros_like(adapted_features[0])
#         for i in range(len(self.adapters)):
#             fused_feature += category_probs[:, i].view(batch_size, 1, 1, 1) * adapted_features[i]
#
#         return fused_feature


import math, torch
from torch import nn
import torch.nn.functional as F

# 定义多尺度局部上下文注意力模块 (MLCA)
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()

        # 初始化参数
        # in_size: 输入的通道数
        # local_size: 局部池化尺寸，默认为5
        # gamma 和 b 用于计算卷积核的大小
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        # 根据 ECA 方法计算卷积核大小 k
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1  # 保证 k 是奇数，以便对称填充
        # 定义两个 1D 卷积，用于全局和局部的注意力计算
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        # 局部和全局注意力的加权参数
        self.local_weight = local_weight
        # 定义自适应平均池化，用于局部和全局特征提取
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 进行局部和全局的自适应平均池化
        local_arv = self.local_arv_pool(x)  # 局部特征池化
        global_arv = self.global_arv_pool(local_arv)  # 从局部特征中进一步提取全局特征
        # 获取输入和池化后的特征的形状
        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape
        # 将局部特征重新排列为 (b, 1, local_size*local_size*c) 以便于通过 1D 卷积
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        # 将全局特征重新排列为 (b, 1, c)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
        # 通过局部卷积计算局部注意力
        y_local = self.conv_local(temp_local)
        # 通过全局卷积计算全局注意力
        y_global = self.conv(temp_global)
        # 将局部注意力重新排列回原始形状 (b, c, local_size, local_size)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c, self.local_size, self.local_size)
        # 将全局注意力重新排列回 (b, c, 1, 1)
        y_global_transpose = y_global.transpose(-1, -2).unsqueeze(-1)
        # 应用 sigmoid 激活函数，将注意力权重映射到 (0, 1) 区间
        att_local = y_local_transpose.sigmoid()
        # 将全局注意力池化到局部特征的大小
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        # 根据局部和全局的加权参数，融合两种注意力，调整到输入的空间维度
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])
        # 将输入特征与注意力权重相乘，得到加权后的输出
        x = x * att_all
        return x

# # 测试代码块
# if __name__ == '__main__':
#     # 创建 MLCA 模块实例，输入通道数为 256
#     attention = MLCA(in_size=256)
#     # 随机生成输入张量，形状为 (2, 256, 16, 16)
#     inputs = torch.randn((2, 256, 16, 16))
#     # 将输入张量传入 MLCA 模块，计算输出
#     result = attention(inputs)
#     # 打印输出张量的形状
#     print(result.size())

import torch
import torch.nn as nn
from torch.nn import init


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)




class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)
        return out

#
# class EnhancedSimpleCBAM(nn.Module):
#     def __init__(self, in_channels=3, feature_dim=None, seq_length=None, num_categories=15):
#         super().__init__()
#
#         # 初始特征提取和通道扩展
#         self.conv_in = nn.Sequential(
#                     nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(64),  # 使用BatchNorm代替InstanceNorm
#                     nn.SiLU()
#                 )
#
#
#         # 局部特征提取模块
#         self.local_feature_extractor = LocalFeatureExtractor(in_channels=64)
#
#         # 多尺度模块
#         self.multi_scale = MultiScaleModule(64)
#         # self.se = SELayer(64)
#
#         # 空间注意力（通道数增加到128）
#         self.sa = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.SiLU(),
#             SimpleSpatialAttention()
#         )
#
#         # 通道注意力（通道数增加到128）
#         self.ca = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=1),
#             nn.BatchNorm2d(128),
#             nn.SiLU(),
#             ChannelAttentionBlock(128)
#         )
#
#         # 降维到64
#         self.reduce_channels = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=1),
#             nn.BatchNorm2d(64),
#             nn.SiLU()
#         )
#         # # 注意力模块
#         # self.ca = ChannelAttentionBlock(64)
#         # self.sa = SimpleSpatialAttention()
#
#         # self.ga = GatedAttentionBlock(128)  # 添加分组门控注意力模块
#
#
#         # 类别特定适配器
#         self.category_specific_adapter = CategorySpecificAdapter(input_dim=64, num_categories=num_categories)
#         # 多尺度局部上下文注意力模块 (MLCA)
#         self.mlca = MLCA(in_size=64)
#
#         self.feature_dim = feature_dim
#         self.seq_length = seq_length
#         if feature_dim and seq_length:
#             self.proj = nn.Sequential(
#                 nn.AdaptiveAvgPool2d((8, 8)),
#                 nn.Flatten(),
#                 nn.Linear(64 * 64, 256),
#                 nn.LayerNorm(256),
#                 nn.SiLU(),
#                 nn.Linear(256, feature_dim)
#             )
#         # self.connect = ContextAwareAdapter(64)
#         # if feature_dim and seq_length:
#         #     self.proj = nn.Sequential(
#         #         nn.AdaptiveAvgPool2d((4, 4)),
#         #         nn.Flatten(),
#         #         nn.Linear(64 * 16, feature_dim),
#         #         nn.LayerNorm(feature_dim),
#         #         nn.SiLU()
#         #     )
#
#     # def forward(self, x):
#     #     identity = x
#     #
#     #     # 初始特征提取
#     #     x_1 = self.conv_in(x)
#     #
#     #     x_1 = self.mlca(x_1)
#     #
#     #     # 局部特征提取
#     #     x_local = self.local_feature_extractor(x_1)
#     #
#     #     # 打印形状以调试
#     #     # print(f"x shape: {x.shape}, x_local shape: {x_local.shape}")
#     #
#     #     # 确保 x 和 x_local 的形状一致
#     #     if x.shape != x_local.shape:
#     #         # 例如，使用卷积调整 x_local 的形状
#     #         x_local = nn.functional.interpolate(x_local, size=x.shape[2:], mode='bilinear', align_corners=False)
#     #
#     #
#     #
#     #     # 多尺度特征增强
#     #     x_2 = self.sa(x_local)
#     #     x_2 = x_2 + x_1 + x_local
#     #
#     #     # 通道注意力
#     #     x_3 = self.ca(x_2)
#     #     x_3 = x_3 + x_2 + x_1 + x_local
#     #
#     #
#     #     # 空间注意力
#     #     x_4 = self.multi_scale(x_3)
#     #     x = x_4 + x_3 + x_2 + x_1 + x_local
#     #
#     #     # 应用类别特定适配器
#     #     x = self.category_specific_adapter(x)
#     #
#     #     x = x + x_4 + x_3
#     #
#     #
#     #
#     #     # x = self.se(x)
#     #
#     #     # x_con = self.connect(x)
#     #     # x = x_con + x_3 + x_2 + x_1 + x_local
#     #
#     #     # 将局部特征与主特征融合
#     #
#     #
#     #     if self.feature_dim and self.seq_length:
#     #         # 投影到目标维度
#     #         features = self.proj(x)
#     #         # 扩展到序列长度
#     #         features = features.unsqueeze(1).expand(-1, self.seq_length, -1)
#     #         return features, features
#     #
#     #     return x # 返回 U-Net 的输出
#
#     def forward(self, x):
#         identity = x
#
#         # 初始特征提取
#         x_1 = self.conv_in(x)
#
#
#
#         # 空间注意力
#         x_2 = self.sa(x_1)
#         # x_2 = torch.cat([x_2, x_1], dim=1)
#
#
#         # 通道注意力
#         x_3 = self.ca(x_2)
#         # x_3 = torch.cat([x_3, x_2, x_1], dim=1)
#
#
#         x_3 = self.reduce_channels(x_3)
#
#
#
#         # 多尺度
#         x_4 = self.multi_scale(x_3)
#         # x_4 = torch.cat([x_4, x_3, x_2, x_1], dim=1)
#         x_4 = x_4 + x_1
#
#         # mlca
#         x_5 = self.mlca(x_4)
#         # x_5 = torch.cat([x_5, x_4, x_3, x_2, x_1], dim=1)
#         x_5 = x_5 + x_1
#
#         # 局部特征提取
#         x_local = self.local_feature_extractor(x_5)
#
#         # 打印形状以调试
#         # print(f"x shape: {x.shape}, x_local shape: {x_local.shape}")
#
#         # 确保 x 和 x_local 的形状一致
#         if x.shape != x_local.shape:
#             # 例如，使用卷积调整 x_local 的形状
#             x_local = nn.functional.interpolate(x_local, size=x.shape[2:], mode='bilinear', align_corners=False)
#
#         x = x_local + x_1 + x_3 + x_4 + x_5
#
#         # 适配器
#         x = self.category_specific_adapter(x)
#
#         # x = self.se(x)
#
#         # x_con = self.connect(x)
#         # x = x_con + x_3 + x_2 + x_1 + x_local
#
#         # 将局部特征与主特征融合
#
#
#         if self.feature_dim and self.seq_length:
#             # 投影到目标维度
#             features = self.proj(x)
#             # 扩展到序列长度
#             features = features.unsqueeze(1).expand(-1, self.seq_length, -1)
#             return features, features
#
#         return x # 返回 U-Net 的输出

#     最完美
class EnhancedSimpleCBAM(nn.Module):
    def __init__(self, in_channels=3, feature_dim=None, seq_length=None, num_categories=15):
        super().__init__()

        # 初始特征提取和通道扩展
        self.conv_in = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),  # 使用BatchNorm代替InstanceNorm
                    nn.SiLU()
                )


        # 局部特征提取模块
        self.local_feature_extractor = LocalFeatureExtractor(in_channels=64)

        # 多尺度模块      6 为mpdd， 12 btad  15visa ,Brain   10mvtec
        self.multi_scale = CategorySpecificAdapter_multi(input_dim=64,  num_categories=12)
        # self.multi_scale = MultiScaleModule(64)
        # self.se = SELayer(64)
        # 注意力模块
        self.ca = ChannelAttentionBlock(64)
        self.sa = SimpleSpatialAttention()

        # self.ga = GatedAttentionBlock(128)  # 添加分组门控注意力模块


        # 类别特定适配器
        self.category_specific_adapter = CategorySpecificAdapter(input_dim=64, num_categories=num_categories)
        # 多尺度局部上下文注意力模块 (MLCA)
        # self.mlca = MLCA(in_size=64)

        self.feature_dim = feature_dim
        self.seq_length = seq_length

        # self.connect = ContextAwareAdapter(64)
        if feature_dim and seq_length:
            self.proj = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 16, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU()
            )

    def forward(self, x):
        identity = x

        # 初始特征提取
        x_1 = self.conv_in(x)

        # x_1 = self.mlca(x_1)

        # 局部特征提取
        x_local = self.local_feature_extractor(x_1)

        # 打印形状以调试
        # print(f"x shape: {x.shape}, x_local shape: {x_local.shape}")

        # 确保 x 和 x_local 的形状一致
        if x.shape != x_local.shape:
            # 例如，使用卷积调整 x_local 的形状
            x_local = nn.functional.interpolate(x_local, size=x.shape[2:], mode='bilinear', align_corners=False)



        # 多尺度特征增强
        x_2 = self.sa(x_local)
        x_2 = x_2 + x_1 + x_local
        #
        # # 通道注意力
        x_3 = self.ca(x_2)
        x_3 = x_3 + x_2 + x_1 + x_local
        #
        #
        # # 空间注意力
        x_4 = self.multi_scale(x_3)
        x = x_4 + x_3 + x_2 + x_1 + x_local
        # #
        # # 应用类别特定适配器
        x = self.category_specific_adapter(x)
        #
        x = x + x_3 + x_4



        # x = self.se(x)

        # x_con = self.connect(x)
        # x = x_con + x_3 + x_2 + x_1 + x_local

        # 将局部特征与主特征融合


        if self.feature_dim and self.seq_length:
            # 投影到目标维度
            features = self.proj(x)
            # 扩展到序列长度
            features = features.unsqueeze(1).expand(-1, self.seq_length, -1)
            return features, features

        return x # 返回 U-Net 的输出


# 功能最好
# class EnhancedSimpleCBAM(nn.Module):
#     def __init__(self, in_channels=3, feature_dim=None, seq_length=None):
#         super().__init__()
#
#         # 初始特征提取和通道扩展
#         self.conv_in = nn.Sequential(
#                     nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#                     nn.BatchNorm2d(64),  # 使用BatchNorm代替InstanceNorm
#                     nn.SiLU()
#                 )
#
#
#         # 局部特征提取模块
#         self.local_feature_extractor = LocalFeatureExtractor(in_channels=64)
#
#         # 多尺度模块
#         self.multi_scale = MultiScaleModule(64)
#         self.connect = ContextAwareAdapter(64)
#         # 注意力模块
#         self.ca = ChannelAttentionBlock(64)
#         self.sa = SimpleSpatialAttention()
#
#         self.ga = GatedAttentionBlock(64)  # 添加分组门控注意力模块
#
#         self.feature_dim = feature_dim
#         self.seq_length = seq_length
#
#         if feature_dim and seq_length:
#             self.proj = nn.Sequential(
#                 nn.AdaptiveAvgPool2d((4, 4)),
#                 nn.Flatten(),
#                 nn.Linear(64 * 16, feature_dim),
#                 nn.LayerNorm(feature_dim),
#                 nn.SiLU()
#             )
#
#     def forward(self, x):
#         identity = x
#
#         # 初始特征提取
#         x_1 = self.conv_in(x)
#
#         # 局部特征提取
#         x_local = self.local_feature_extractor(x_1)
#
#         # 打印形状以调试
#         # print(f"x shape: {x.shape}, x_local shape: {x_local.shape}")
#
#         # 确保 x 和 x_local 的形状一致
#         if x.shape != x_local.shape:
#             # 例如，使用卷积调整 x_local 的形状
#             x_local = nn.functional.interpolate(x_local, size=x.shape[2:], mode='bilinear', align_corners=False)
#
#         # 多尺度特征增强
#         x_2 = self.sa(x_local)
#         x_2 = x_2 + x_1 + x_local
#
#         # 通道注意力
#         x_3 = self.ca(x_2)
#         x_3 = x_3 + x_2 + x_1 + x_local
#
#         x_con = self.connect(x_3)
#         x_con = x_con + x_3 + x_2 + x_1 + x_local
#         # 空间注意力
#         x_4 = self.multi_scale(x_con)
#         x = x_4 + x_con + x_3 + x_2 + x_1 + x_local
#         #
#         # # 应用分组门控注意力
#         # x = self.ga(x)
#
#         # 将局部特征与主特征融合
#
#
#         if self.feature_dim and self.seq_length:
#             # 投影到目标维度
#             features = self.proj(x)
#             # 扩展到序列长度
#             features = features.unsqueeze(1).expand(-1, self.seq_length, -1)
#             return features, features
#
#         return x # 返回 U-Net 的输出



# class EnhancedSimpleCBAM(nn.Module):
#     def __init__(self, in_channels=3, feature_dim=None, seq_length=None):
#         super().__init__()
#
#         # 初始特征提取和通道扩展
#         self.conv_in = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),  # 使用BatchNorm代替InstanceNorm
#             nn.SiLU()
#         )
#
#         # 多尺度模块
#         self.multi_scale = MultiScaleModule(64)
#
#         # 注意力模块
#         self.ca = ChannelAttentionBlock(64)
#         self.sa = SimpleSpatialAttention()
#
#         self.feature_dim = feature_dim
#         self.seq_length = seq_length
#
#         if feature_dim and seq_length:
#             self.proj = nn.Sequential(
#                 nn.AdaptiveAvgPool2d((4, 4)),
#                 nn.Flatten(),
#                 nn.Linear(64 * 16, feature_dim),
#                 nn.LayerNorm(feature_dim),
#                 nn.SiLU()
#             )
#
#     def forward(self, x):
#         identity = x
#
#         # 初始特征提取
#         x_1 = self.conv_in(x)
#
#         # 多尺度特征增强
#         x_2 = self.sa(x_1)
#         x_2 = x_2 + x_1
#
#         # 通道注意力
#         x_3 = self.ca(x_2)
#         x_3 = x_3 + x_2 + x_1
#
#         # 空间注意力
#         x_4 = self.multi_scale(x_3)
#         x = x_4 + x_3 + x_2 + x_1
#
#         if self.feature_dim and self.seq_length:
#             # 投影到目标维度
#             features = self.proj(x)
#             # 扩展到序列长度
#             features = features.unsqueeze(1).expand(-1, self.seq_length, -1)
#             return features, features
#
#         return x


import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 中间层
        self.bottleneck = self.conv_block(512, 1024)

        # 解码器部分
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        # 中间层
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))

        # 解码
        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # 跳跃连接
        dec4 = self.conv_block(1024, 512)(dec4)

        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 跳跃连接
        dec3 = self.conv_block(512, 256)(dec3)

        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 跳跃连接
        dec2 = self.conv_block(256, 128)(dec2)

        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 跳跃连接
        dec1 = self.conv_block(128, 64)(dec1)

        return self.final_conv(dec1)  # 输出层



class EnhancedImageProcessor(nn.Module):
    def __init__(self, in_channels=3):
        (super().
         __init__())

        # 保持输入输出通道数一致
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU()
        )

        # 多尺度模块 - 保持通道数不变
        self.multi_scale = MultiScaleModule(in_channels)

        # 注意力模块 - 使用输入通道数
        # self.ca = SimpleChannelAttention(in_channels)
        self.sa = SimpleSpatialAttention()

        # 最终输出调整
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU()
        )

        # 残差连接权重
        self.alpha = 0.85

    def forward(self, x):
        identity = x

        # 特征增强
        x_1 = self.multi_scale(x)
        x_1 = x_1 + x

        x_2 = self.conv_in(x_1)
        x_2 = x_2 + x_1

        x_3 = self.sa( x_2)
        x_3 = x_3 + x_2

        x_4 = self.ca(x_3)
        x_4 = x_4 + x_3

        x_5 = self.conv_out(x_4)
        x_5 = x_5 + x_4
        # 残差连接
        # output = self.alpha * identity + (1 - self.alpha) * x
        output = identity + x_5
        return output


import torch
import torch.nn as nn

# 定义空间位移函数
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x

def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x

class SplitAttention(nn.Module):
    def __init__(self, channel, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # bs, k, n, c
        a = torch.sum(torch.sum(x_all, dim=1), dim=1)  # bs, c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs, kc
        hat_a = hat_a.reshape(b, self.k, c)  # bs, k, c
        bar_a = self.softmax(hat_a)  # bs, k, c
        attention = bar_a.unsqueeze(-2)  # bs, k, 1, c
        out = attention * x_all  # bs, k, n, c
        out = torch.sum(out, dim=1).reshape(b, h, w, c)  # bs, h, w, c
        return out

class S2Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)  # Ensure channels match your input
        self.mlp2 = nn.Linear(channels * 3, channels)  # Ensure output channels match your needs
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)  # Change to (b, h, w, c)

        # Ensure the input to mlp1 has the correct shape
        x = x.view(b, -1, c)  # Reshape to (b, h*w, c)
        x = self.mlp1(x)  # Now x should have shape (b, h*w, channels*3)

        # Split the channels for spatial shifts
        x1 = spatial_shift1(x[:, :, :c])
        x2 = spatial_shift2(x[:, :, c:c * 2])
        x3 = x[:, :, c * 2:]

        x_all = torch.stack([x1, x2, x3], dim=1)  # (b, 3, h*w, c)
        a = self.split_attention(x_all)  # Process through split attention
        x = self.mlp2(a)  # Final linear transformation
        x = x.permute(0, 3, 1, 2)  # Change back to (b, c, h, w)
        return x


import torch
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.init as init

class PSA(nn.Module):
    def __init__(self, channel=3, reduction=4, S=1):  # Set S to 1 for RGB images
        super().__init__()
        if channel <= 0 or S <= 0 or channel % S != 0:
            raise ValueError(
                f"Invalid parameters: channel={channel}, S={S}. Channel must be a positive integer divisible by S.")

        self.S = S
        self.channel_per_slice = channel // S
        if self.channel_per_slice <= 0:
            raise ValueError(f"Channel per slice {self.channel_per_slice} must be greater than 0.")

        self.convs = nn.ModuleList([
            nn.Conv2d(self.channel_per_slice, self.channel_per_slice, kernel_size=2 * (i + 1) + 1, padding=i + 1) for i
            in range(S)
        ])
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.channel_per_slice, self.channel_per_slice // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channel_per_slice // reduction, self.channel_per_slice, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for i in range(S)
        ])
        self.softmax = nn.Softmax(dim=1)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.size(0) > 0:  # 确保权重张量不是零元素
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                else:
                    print(f"Warning: Skipping weight initialization for Conv2d layer with zero-element tensor.")
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        if c % self.S != 0:
            raise ValueError(f"Input channels {c} must be divisible by S={self.S}.")

        print(f"Input shape: {x.shape}")  # 调试信息

        # Step 1: SPC module
        SPC_out = x.view(b, self.S, c // self.S, h, w)  # bs, s, ci, h, w
        for idx, conv in enumerate(self.convs):
            print(f"SPC processing slice {idx}, input shape: {SPC_out[:, idx, :, :, :].shape}")  # 调试信息
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        # Step 2: SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            print(f"SE block {idx} processing, input shape: {SPC_out[:, idx, :, :, :].shape}")  # 调试信息
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step 3: Softmax
        softmax_out = self.softmax(SE_out)

        # Step 4: SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)  # Flatten the output
        print(f"Output shape: {PSA_out.shape}")  # 调试信息
        return PSA_out




# 测试代码
# if __name__ == "__main__":
#     model = PSA(channel=3, reduction=4, S=1)
#     input_tensor = torch.randn(2, 3, 64, 64)
#     output_tensor = model(input_tensor)
#     print(output_tensor.shape)
# import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

# 论文地址：https://arxiv.org/pdf/1910.03151
# 论文：ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.0001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs, c, 1, 1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs, 1, c
        y = self.conv(y)  # bs, 1, c
        y = self.relu(y)  #Apply ReLU
        y = self.sigmoid(y)  # bs, 1, c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs, c, 1, 1
        return x * y.expand_as(x)

if __name__ == '__main__':
    input_tensor = torch.randn(50, 512, 7, 7)
    block = ECAAttention(kernel_size=3)
    output = block(input_tensor)
    print(output.shape)
