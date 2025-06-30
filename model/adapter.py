"""
  @Author: 王权
  @FileName: adapter.py
  @DateTime: 2025/4/2 21:55
  @SoftWare: PyCharm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# class ContextAwareAdapter(nn.Module):
#     def __init__(self, input_dim):
#         super(ContextAwareAdapter, self).__init__()
#         self.fc = nn.Linear(input_dim, input_dim)
#         self.norm = nn.LayerNorm(input_dim)
#         self.dropout = nn.Dropout(0.1)
#         self.attention = nn.MultiheadAttention(input_dim, num_heads=8, dropout=0.1)
#
#     def forward(self, x):
#         # 假设 x 的形状为 [height, batch_size, channels]
#         # 转换为 [batch_size, height, channels]
#         x = x.permute(1, 0, 2)
#         # 应用注意力机制
#         attn_output, _ = self.attention(x, x, x)
#         x = x + attn_output  # 残差连接
#         x = self.fc(x)
#         x = self.norm(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         # 转换回原始形状 [height, batch_size, channels]
#         x = x.permute(1, 0, 2)
#         return x

# 基本的最初的
# class ContextAwareAdapter(nn.Module):
#     def __init__(self, input_dim):
#         super(ContextAwareAdapter, self).__init__()
#         self.fc = nn.Linear(input_dim, input_dim)
#         self.norm = nn.LayerNorm(input_dim)
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         # 假设 x 的形状为 [height, batch_size, channels]
#         # 转换为 [batch_size, height, channels]
#         x = x.permute(1, 0, 2)
#         x = self.fc(x)
#         x = self.norm(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         # 转换回原始形状 [height, batch_size, channels]
#         x = x.permute(1, 0, 2)
#         return x

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
#         # 假设 x 的形状为 [height, batch_size, channels]
#         height, batch_size, channels = x.shape
#
#         # 获取类别概率
#         # 对特征进行全局平均池化，得到 [batch_size, channels]
#         global_features = x.mean(dim=0)
#         category_probs = self.category_classifier(global_features)
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
#             # 将类别概率扩展到与特征图相同的形状
#             category_prob = category_probs[:, i].view(batch_size, 1, 1)
#             fused_feature += category_prob * adapted_features[i]
#
#         return fused_feature
#



class CategorySpecificAdapter(nn.Module):
    def __init__(self, input_dim, num_categories):
        super().__init__()
        self.adapters = nn.ModuleList([CBAM(input_dim) for _ in range(num_categories)])
        self.category_classifier = nn.Sequential(
            nn.Linear(input_dim, num_categories),
            nn.Softmax(dim=1)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # 假设 x 的形状为 [height, batch_size, channels]
        height, batch_size, channels = x.shape

        # 获取类别概率
        # 对特征进行全局平均池化，得到 [batch_size, channels]
        global_features = x.mean(dim=0)
        category_probs = self.category_classifier(global_features)

        # 应用类别特定适配器
        adapted_features = []
        for adapter in self.adapters:
            adapted = adapter(x)
            adapted_features.append(adapted)

        # 根据类别概率融合特征
        fused_feature = torch.zeros_like(adapted_features[0])
        for i in range(len(self.adapters)):
            # 将类别概率扩展到与特征图相同的形状
            category_prob = category_probs[:, i].view(batch_size, 1, 1)
            fused_feature += category_prob * adapted_features[i]

        # 对融合后的特征进行归一化
        fused_feature = self.norm(fused_feature.permute(1, 0, 2)).permute(1, 0, 2)
        return fused_feature

import torch.nn as nn
# 可以的
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f"ChannelAttention input shape: {x.shape}")  # 打印输入形状

        avg_out = self.avg_pool(x)  # 输出形状为 (batch_size, in_channels, 1, 1)
        # print(f"ChannelAttention avg_out shape: {avg_out.shape}")  # 打印平均池化输出形状

        avg_out = self.fc(avg_out)  # 通过全连接层
        # print(f"ChannelAttention avg_out after fc shape: {avg_out.shape}")  # 打印经过全连接层后的形状

        max_out = self.max_pool(x)  # 输出形状为 (batch_size, in_channels, 1, 1)
        # print(f"ChannelAttention max_out shape: {max_out.shape}")  # 打印最大池化输出形状

        max_out = self.fc(max_out)  # 通过全连接层
        # print(f"ChannelAttention max_out after fc shape: {max_out.shape}")  # 打印经过全连接层后的形状

        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(f"SpatialAttention input shape: {x.shape}")  # 打印输入形状

        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(f"SpatialAttention avg_out shape: {avg_out.shape}")  # 打印平均池化输出形状

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(f"SpatialAttention max_out shape: {max_out.shape}")  # 打印最大池化输出形状

        x = torch.cat([avg_out, max_out], dim=1)
        # print(f"SpatialAttention concatenated shape: {x.shape}")  # 打印拼接后的形状

        x = self.conv(x)
        # print(f"SpatialAttention output shape: {x.shape}")  # 打印输出形状

        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)



    def forward(self, x):
        # print(f"CBAM input shape: {x.shape}")  # 打印输入

        x = x.permute(1, 2, 0)
        x = x.unsqueeze(3)  # 添加一个维度，变为 [batch_size, channels, seq_length,  1]
        # print(f"CBAM input after permute shape: {x.shape}")
        out_chan = self.channel_attention(x) * x
        # print(f"CBAM after channel attention shape: {out.shape}")  # 打印通道注意力后的形状

        out_spa = self.spatial_attention(x) * x
        out = out_chan * out_spa
        # print(f"CBAM output shape: {out.shape}")  # 打印最终输出形状
        out = out.squeeze(-1)  # 或者 x.squeeze(dim=3)
        out = out.permute(2, 0, 1)
        # print( f"CBAM output shape: {out}")
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        初始化SELayer类。
        参数:
        channel (int): 输入特征图的通道数。
        reduction (int): 用于减少通道数的缩减率，默认为16。它用于在全连接层中压缩特征的维度。
        """
        super(SELayer, self).__init__()
        # 自适应平均池化层，将每个通道的空间维度（H, W）压缩到1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层序列，包含两个线性变换和中间的ReLU激活函数
        self.fc = nn.Sequential(
            # 第一个线性层，将通道数从 'channel' 缩减到 'channel // reduction'
            nn.Linear(channel, channel // reduction, bias=False),
            # ReLU激活函数，用于引入非线性
            nn.ReLU(inplace=True),
            # 第二个线性层，将通道数从 'channel // reduction' 恢复到 'channel'
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid激活函数，将输出限制在(0, 1)之间
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播函数。
        参数:
        x (Tensor): 输入张量，形状为 (batch_size, channel, height, width)。
        返回:
        Tensor: 经过通道注意力调整后的输出张量，形状与输入相同。
        """
        # 获取输入张量的形状
        b, c, h, w = x.size()
        # Squeeze：通过全局平均池化层，将每个通道的空间维度（H, W）压缩到1x1
        y = self.avg_pool(x).view(b, c)
        # Excitation：通过全连接层序列，对压缩后的特征进行处理
        y = self.fc(y).view(b, c, 1, 1)
        # 通过扩展后的注意力权重 y 调整输入张量 x 的每个通道
        return x * y.expand_as(x)
import torch




# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         print(f"ChannelAttention input shape: {x.shape}")  # 打印输入形状
#
#         # 平均池化和最大池化的输出
#         avg_out = self.avg_pool(x)  # 输出形状为 (batch_size, in_channels, 1, 1)
#         print(f"ChannelAttention avg_out shape: {avg_out.shape}")
#         avg_out = self.fc(avg_out)  # 通过全连接层
#         print(f"ChannelAttention avg_out shape: {avg_out.shape}")  # 打印平均池化输出形状
#
#         max_out = self.max_pool(x)  # 输出形状为 (batch_size, in_channels, 1, 1)
#         max_out = self.fc(max_out)  # 通过全连接层
#         print(f"ChannelAttention max_out shape: {max_out.shape}")  # 打印最大池化输出形状
#
#         # 将两个输出相加
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         print(f"SpatialAttention input shape: {x.shape}")  # 打印输入形状
#
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         print(f"SpatialAttention avg_out shape: {avg_out.shape}")  # 打印平均池化输出形状
#
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         print(f"SpatialAttention max_out shape: {max_out.shape}")  # 打印最大池化输出形状
#
#         x = torch.cat([avg_out, max_out], dim=1)
#         print(f"SpatialAttention x shape: {x.shape}")
#         x = x.permute(1, 0, 2)
#         print(f"SpatialAttention x shape after permute: {x.shape}")
#         x = self.conv(x)
#         print(f"SpatialAttention output shape: {x.shape}")  # 打印输出形状
#
#         return self.sigmoid(x)
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
#         self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
#
#         # 添加一个线性层来调整通道数
#         self.adjust_channels = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         print(f"CBAM input shape: {x.shape}")  # 打印输入形状
#
#         # 调整通道数
#         x = x.permute(2, 1, 0)
#         print(f"CBAM input shape after permute: {x.shape}")
#         out = self.channel_attention(x) * x
#         print(f"CBAM after channel attention shape: {out.shape}")  # 打印通道注意力后的形状
#
#         out = self.spatial_attention(out) * out
#         print(f"CBAM output shape: {out.shape}")  # 打印最终输出形状
#         out = out.view_as(x)  # 调整输出形状以匹配输入形状
#         print(f"CBAM output shape after view_as: {out.shape}")
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert (
            self.head_dim * num_heads == dim
        ), "dim must be divisible by num_heads"

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # 线性变换
        q = self.linear_q(x)  # [batch_size, seq_length, dim]
        k = self.linear_k(x)  # [batch_size, seq_length, dim]
        v = self.linear_v(x)  # [batch_size, seq_length, dim]

        # 分头
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_length, seq_length]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_length, seq_length]

        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_length, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)  # [batch_size, seq_length, dim]

        # 线性变换和残差连接
        output = self.linear_out(attn_output) + x  # 残差连接
        output = self.layer_norm(output)  # 归一化

        return output

# class AttentionModule(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(dim, dim // 4),
#             nn.LayerNorm(dim // 4),
#             nn.GELU(),
#             nn.Linear(dim // 4, dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.attention(x)
import torch
import torch.nn as nn

class ContextAwareAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.context_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        # 打印输入形状
        # print(f"Input shape: {x.shape}")

        # 计算全局上下文信息，输出形状为 (batch_size, channels)
        context_info = x.mean(dim=1)  # 计算全局上下文信息，输出形状为 (batch_size, channels)

        # 打印上下文信息的形状
        # print(f"Context info shape: {context_info.shape}")

        # 通过上下文层
        context_enhanced = self.context_layer(context_info)  # 形状为 (batch_size, channels)

        # 打印上下文增强的形状
        # print(f"Context enhanced shape: {context_enhanced.shape}")

        # 将上下文信息扩展到 (batch_size, channels, 1, 1)
        context_enhanced = context_enhanced.unsqueeze(1)  # 扩展维度，形状为 (batch_size, 1, channels)
        # print(f"Context enhanced shape after unsqueeze: {context_enhanced.shape}")
        # 返回增强后的特征
        return x + context_enhanced  # 直接相加

class FeatureEnhancement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enhance = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.enhance(x)

    # if not hasattr(self, 'adapters'):
    #     self.adapters = nn.ModuleList([
    #         nn.Sequential(
    #             nn.Linear(x.shape[-1], 768, bias=False),
    #             nn.LayerNorm(768),
    #             nn.GELU(),
    #             nn.Linear(768, x.shape[-1], bias=False)
    #         ) for _ in range(len(self.transformer.resblocks))
    #     ])

# class ChannelAttentionBlock(nn.Module):
#     # 实验不同的超参数：尝试不同的 reduction 值，以找到最佳的通道压缩比例。
#     def __init__(self, in_channels, reduction=16):
#         super(ChannelAttentionBlock, self).__init__()
#
#         # 左侧分支
#         self.left_branch = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # 自适应平均池化
#             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),  # 1x1 卷积
#             nn.ReLU(inplace=True),  # ReLU 激活
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)  # 1x1 卷积
#         )
#
#         # 右侧分支
#         self.right_branch = nn.Sequential(
#             nn.AdaptiveMaxPool2d(1),  # 自适应平均池化
#             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),  # 1x1 卷积
#             nn.ReLU(inplace=True),  # ReLU 激活
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)  # 1x1 卷积
#         )
#
#         # Sigmoid 激活
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         print(f"Input shape: {x.shape}")
#         # 左侧分支输出
#         left_out = self.left_branch(x)
#         print(f"Left branch output shape: {left_out.shape}")
#         # 右侧分支输出
#         right_out = self.right_branch(x)
#         print(f"Right branch output shape: {right_out.shape}")
#         # 融合与激活
#         out = left_out + right_out  # 加法操作
#         print(f"Output shape: {out.shape}")
#         attention = self.sigmoid(out)  # Sigmoid 激活
#         print( f"Attention shape: {attention.shape}")
#         # 输出
#         return x * attention  # 通道注意力权重应用
#
# class SimpleSpatialAttention(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(16)  # 添加 Batch Normalization
#             self.relu = nn.ReLU(inplace=True)  # 使用 ReLU 激活
#             self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False)
#             self.bn2 = nn.BatchNorm2d(1)  # 添加 Batch Normalization
#             self.sigmoid = nn.Sigmoid()
#
#         def forward(self, x):
#             avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算平均池化
#             max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算最大池化
#             x = torch.cat([avg_out, max_out], dim=1)  # 拼接平均和最大池化结果
#             x = x.permute(1, 0, 2)
#             print( "x.shape ",x.shape)
#             x = self.conv1(x)  # 第一个卷积层
#             x = x.unsqueeze(3)  # 添加一个维度，变为 [batch_size, seq_length, channels, 1]
#             x = x.permute(3, 0, 1, 2)  # 变换为 [batch_size, channels, seq_length, 1]
#             print("xconv1d.shape ", x.shape)
#             x = self.bn1(x)  # Batch Normalization
#             print( "xbn1d.shape ",x.shape)
#             x = self.relu(x)  # 激活函数
#             print( "xrelud.shape ",x.shape)
#             x = self.conv2(x)  # 第二个卷积层
#             print( "xconv2d.shape ",x.shape)
#             x = self.bn2(x)  # Batch Normalization
#             print( "xbn2d.shape ",x.shape)
#             attention = self.sigmoid(x)  # Sigmoid 激活
#             print( "attention.shape ",attention.shape)
#             return x * attention  # 应用注意力权重