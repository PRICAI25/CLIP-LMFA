"""
  @Author: 王权
  @FileName: gem_utils.py
  @DateTime: 2025/3/19 13:21
  @SoftWare: PyCharm
"""
from typing import Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from .adapter import CategorySpecificAdapter, CBAM
from .transformer import _expand_token, to_2tuple



def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True
):
    # sort out sizes, assume square if old size not provided
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    if new_size == old_size:  # might not both be same container type
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)


    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb

class SelfSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ss_attn_iter=1,
                 ss_attn_temp=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.ss_attn_iter = ss_attn_iter
        self.ss_attn_temp = ss_attn_temp

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None, prev_attn=None):
        x = x.transpose(0, 1)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        self.v_values = v
        # original self-attention for the original path
        attn_ori_return = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori_return.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x_ori = self.proj_drop(self.proj(x_ori))

        # GEM
        xs1 = v
        xs2 = k
        xs3 = q

        if self.ss_attn_temp is None:
            pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
            inv_temp = pre_norm * self.scale
        else:
            inv_temp = self.ss_attn_temp

        for it in range(self.ss_attn_iter):
            xs1 = F.normalize(xs1, dim=-1)
            xs2 = F.normalize(xs2, dim=-1)
            xs3 = F.normalize(xs3, dim=-1)

            attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
            attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
            attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

            attn1 = (attn_return1).softmax(dim=-1)
            attn2 = (attn_return2).softmax(dim=-1)
            attn3 = (attn_return3).softmax(dim=-1)

            xs1 = attn1 @ xs1
            xs2 = attn2 @ xs2
            xs3 = attn3 @ xs3

        # Assigment to V
        xs1 = F.normalize(xs1, dim=-1)
        xs2 = F.normalize(xs2, dim=-1)
        xs3 = F.normalize(xs3, dim=-1)

        attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
        attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
        attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

        attn1 = (attn_return1).softmax(dim=-1)
        attn2 = (attn_return2).softmax(dim=-1)
        attn3 = (attn_return3).softmax(dim=-1)

        xs1 = attn1 @ v
        xs2 = attn2 @ v
        xs3 = attn3 @ v
        xs = (xs1 + xs2 + xs3) / 3

        x = xs.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        return [x.transpose(0, 1), x_ori.transpose(0, 1)]


# 修改1
# class SelfSelfAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ss_attn_iter=1,
#                  ss_attn_temp=None):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.ss_attn_iter = ss_attn_iter
#         self.ss_attn_temp = nn.Parameter(torch.tensor(1.0)) if ss_attn_temp is None else ss_attn_temp
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.ln = nn.LayerNorm(dim)  # 添加层归一化提升稳定性
#
#     def forward(self, x, attn_bias=None, prev_attn=None):
#         x = x.transpose(0, 1)
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)  # 使用unbind替代索引，更高效
#         self.v_values = v
#
#         # 原始自注意力
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = self.attn_drop(attn.softmax(dim=-1))
#         x_ori = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x_ori = self.proj_drop(self.proj(x_ori))
#
#         # GEM优化
#         xs = [v, k, q]  # 初始化特征组
#         inv_temp = self.ss_attn_temp * self.scale  # 动态温度
#
#         for _ in range(self.ss_attn_iter):
#             attns = []
#             for x_feat in xs:
#                 x_norm = F.normalize(x_feat, dim=-1)
#                 attn = (x_norm @ x_norm.transpose(-2, -1)) * inv_temp
#                 attns.append(self.attn_drop(attn.softmax(dim=-1)))
#
#             xs = [attn @ x for attn, x in zip(attns, xs)]  # 更新特征
#
#         # 融合与投影
#         xs_fused = sum(attn @ v for attn in attns) / len(attns)  # 动态加权融合
#         x = xs_fused.transpose(1, 2).reshape(B, N, C)
#
#         # 确保所有张量都在同一个设备上
#         device = x.device
#         self.proj.to(device)
#         self.proj_drop.to(device)
#         self.ln.to(device)
#
#         x = self.proj_drop(self.proj(self.ln(x)))  # 添加层归一化
#
#         return [x.transpose(0, 1), x_ori.transpose(0, 1)]


# 修改2
# class SelfSelfAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ss_attn_iter=1,
#                  ss_attn_temp=None, mlp_ratio=2):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.ss_attn_iter = ss_attn_iter
#         self.ss_attn_temp = ss_attn_temp
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         # 修改MLP模块，减小隐藏层维度和非线性变换
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * mlp_ratio),
#             nn.ReLU(),  # 使用线性激活函数
#             nn.Dropout(proj_drop),
#             nn.Linear(dim * mlp_ratio, dim),
#             nn.Dropout(proj_drop)
#         )
#
#     def forward(self, x, attn_bias=None, prev_attn=None):
#         x = x.transpose(0, 1)
#         device = x.device
#         x = x.to(device)
#         # 确保模型在正确设备上
#         self.mlp = self.mlp.to(device)
#
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         self.v_values = v
#
#         # 原始自注意力
#         attn_ori_return = (q @ k.transpose(-2, -1)) * self.scale
#         attn_ori = attn_ori_return.softmax(dim=-1)
#         attn_ori = self.attn_drop(attn_ori)
#
#         x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
#         x_ori = self.proj_drop(self.proj(x_ori))
#
#         # GEM
#         xs1 = v
#         xs2 = k
#         xs3 = q
#
#         if self.ss_attn_temp is None:
#             pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
#             inv_temp = pre_norm * self.scale
#         else:
#             inv_temp = self.ss_attn_temp
#
#         for it in range(self.ss_attn_iter):
#             xs1 = F.normalize(xs1, dim=-1)
#             xs2 = F.normalize(xs2, dim=-1)
#             xs3 = F.normalize(xs3, dim=-1)
#
#             attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
#             attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
#             attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp
#
#             attn1 = (attn_return1).softmax(dim=-1)
#             attn2 = (attn_return2).softmax(dim=-1)
#             attn3 = (attn_return3).softmax(dim=-1)
#
#             xs1 = attn1 @ xs1
#             xs2 = attn2 @ xs2
#             xs3 = attn3 @ xs3
#
#         # 最终计算
#         xs1 = F.normalize(xs1, dim=-1)
#         xs2 = F.normalize(xs2, dim=-1)
#         xs3 = F.normalize(xs3, dim=-1)
#
#         attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
#         attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
#         attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp
#
#         attn1 = (attn_return1).softmax(dim=-1)
#         attn2 = (attn_return2).softmax(dim=-1)
#         attn3 = (attn_return3).softmax(dim=-1)
#
#         xs1 = attn1 @ v
#         xs2 = attn2 @ v
#         xs3 = attn3 @ v
#         xs = (xs1 + xs2 + xs3) / 3
#
#         x = xs.transpose(1, 2).reshape(B, N, C)
#         x = self.proj_drop(self.proj(x))
#
#
#         # 通过MLP进一步处理特征
#         x = self.mlp(x)
#
#         return [x.transpose(0, 1), x_ori.transpose(0, 1)]

# 修改3
import torch
import torch.nn as nn
import torch.nn.functional as F

# class DilateAttention(nn.Module):
#     def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
#         super().__init__()
#         self.head_dim = head_dim
#         self.scale = qk_scale or head_dim ** -0.5
#         self.kernel_size = kernel_size
#         self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
#         self.attn_drop = nn.Dropout(attn_drop)
#
#     def forward(self, q, k, v):
#         B, d, H, W = q.shape
#         q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
#         k = self.unfold(k).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2, 3)  # B,h,N,d,k*k
#         attn = (q @ k) * self.scale  # B,h,N,1,k*k
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         v = self.unfold(v).reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
#         x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
#         return x
#
# class MultiDilateLocalAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
#                  attn_drop=0., proj_drop=0., kernel_size=3, dilation=[2, 3]):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_dilation = len(dilation)
#         assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
#         self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
#         self.dilate_attention = nn.ModuleList(
#             [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
#              for i in range(self.num_dilation)])
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         print("x before.shape: ",x.shape)
#         x = x.permute(0, 3, 2, 1)
#         print("x after .shape: ", x.shape)
#         B, H, W, C = x.shape
#         x = x.permute(0, 3, 1, 2)
#         qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
#
#         x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
#
#         for i in range(self.num_dilation):
#             x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W, C//num_dilation
#         x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# class SelfSelfAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ss_attn_iter=1,
#                  ss_attn_temp=None, mlp_ratio=4, kernel_size=3, dilation=[2, 3]):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.ss_attn_iter = ss_attn_iter
#         self.ss_attn_temp = ss_attn_temp
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         # 添加多尺度膨胀注意力
#         self.dilate_attention = MultiDilateLocalAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, kernel_size, dilation)
#
#         # 添加MLP模块
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(proj_drop),
#             nn.Linear(dim * mlp_ratio, dim),
#             nn.Dropout(proj_drop)
#         )
#
#     def forward(self, x, attn_bias=None, prev_attn=None):
#         x = x.transpose(0, 1)
#         device = x.device
#         x = x.to(device)
#         self.mlp = self.mlp.to(device)
#         self.dilate_attention = self.dilate_attention.to(device)
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         self.v_values = v
#
#         # 原始自注意力
#         attn_ori_return = (q @ k.transpose(-2, -1)) * self.scale
#         attn_ori = attn_ori_return.softmax(dim=-1)
#         attn_ori = self.attn_drop(attn_ori)
#
#         x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
#         x_ori = self.proj_drop(self.proj(x_ori))
#
#         # GEM
#         xs1 = v
#         xs2 = k
#         xs3 = q
#
#         if self.ss_attn_temp is None:
#             pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
#             inv_temp = pre_norm * self.scale
#         else:
#             inv_temp = self.ss_attn_temp
#
#         for it in range(self.ss_attn_iter):
#             xs1 = F.normalize(xs1, dim=-1)
#             xs2 = F.normalize(xs2, dim=-1)
#             xs3 = F.normalize(xs3, dim=-1)
#
#             attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
#             attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
#             attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp
#
#             attn1 = (attn_return1).softmax(dim=-1)
#             attn2 = (attn_return2).softmax(dim=-1)
#             attn3 = (attn_return3).softmax(dim=-1)
#
#             xs1 = attn1 @ xs1
#             xs2 = attn2 @ xs2
#             xs3 = attn3 @ xs3
#
#         # 最终计算
#         xs1 = F.normalize(xs1, dim=-1)
#         xs2 = F.normalize(xs2, dim=-1)
#         xs3 = F.normalize(xs3, dim=-1)
#
#         attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
#         attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
#         attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp
#
#         attn1 = (attn_return1).softmax(dim=-1)
#         attn2 = (attn_return2).softmax(dim=-1)
#         attn3 = (attn_return3).softmax(dim=-1)
#
#         xs1 = attn1 @ v
#         xs2 = attn2 @ v
#         xs3 = attn3 @ v
#         xs = (xs1 + xs2 + xs3) / 3
#
#         x = xs.transpose(1, 2).reshape(B, N, C)
#         x = self.proj_drop(self.proj(x))
#         print("x before di .shape: ", x.shape)
#         # 通过多尺度膨胀注意力进一步处理特征
#         x = x.permute(0, 2, 1).unsqueeze(-1)  # 调整形状以适应膨胀注意力
#
#         x = self.dilate_attention(x)
#         x = x.permute(0, 2, 3, 1)  # 调整回原始形状
#
#         # 通过MLP进一步处理特征
#         # x = self.mlp(x)
#
#         return [x.transpose(0, 1), x_ori.transpose(0, 1)]

# 修改4
import torch
import torch.nn as nn
import torch.nn.functional as F

# class GatedAttention(nn.Module):
#     def __init__(self, dim, num_experts=4, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_experts = num_experts
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.gate = nn.Linear(dim, num_experts)
#         self.expert_qkv = nn.ModuleList([nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in range(num_experts)])
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         gate_logits = self.gate(x)  # (B, N, num_experts)
#         gate_weights = F.softmax(gate_logits, dim=-1)  # (B, N, num_experts)
#
#         x_list = []
#         for i in range(self.num_experts):
#             expert_qkv = self.expert_qkv[i](x)
#             q, k, v = expert_qkv.chunk(3, dim=-1)
#             q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#             k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#             v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#
#             x_expert = (attn @ v).transpose(1, 2).reshape(B, N, C)
#             x_list.append(x_expert)
#
#         x = torch.stack(x_list, dim=-1)  # (B, N, C, num_experts)
#         x = (x * gate_weights.unsqueeze(2)).sum(dim=-1)  # (B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# class ChunkAttention(nn.Module):
#     def __init__(self, dim, chunk_size=2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.chunk_size = chunk_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         # 确保N能够被chunk_size整除
#         if N % self.chunk_size != 0:
#             # 计算需要填充的长度
#             pad_length = self.chunk_size - (N % self.chunk_size)
#             # 对输入特征进行填充
#             x = F.pad(x, (0, 0, 0, pad_length))
#             N = x.shape[1]
#
#         num_chunks = N // self.chunk_size
#         x = x.reshape(B, num_chunks, self.chunk_size, C)
#
#         qkv = self.qkv(x).reshape(B, num_chunks, self.chunk_size, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 2, 4, 5)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).permute(0, 2, 1, 3, 4).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# class SelfSelfAttention(nn.Module):
#     def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ss_attn_iter=1,
#                  ss_attn_temp=None, mlp_ratio=8, num_experts=4, chunk_size=64):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.ss_attn_iter = ss_attn_iter
#         self.ss_attn_temp = ss_attn_temp
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         # 添加动态路由和分层注意力
#         self.gated_attention = GatedAttention(dim, num_experts, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
#         # self.chunk_attention = ChunkAttention(dim, chunk_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
#
#         # 修改1
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * mlp_ratio),
#             nn.GELU(),
#             nn.Dropout(proj_drop),
#             nn.Linear(dim * mlp_ratio, dim * mlp_ratio // 2),  # 增加一层
#             nn.GELU(),
#             nn.Dropout(proj_drop),
#             nn.Linear(dim * mlp_ratio // 2, dim),
#             nn.Dropout(proj_drop)
#         )
#         # # # 添加MLP模块
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(dim, dim * mlp_ratio),
#         #     nn.GELU(),
#         #     nn.Dropout(proj_drop),
#         #     nn.Linear(dim * mlp_ratio, dim),
#         #     nn.Dropout(proj_drop)
#         # )
#
#     def forward(self, x, attn_bias=None, prev_attn=None):
#         x = x.transpose(0, 1)
#         device = x.device
#         x = x.to(device)
#         self.mlp = self.mlp.to(device)
#         self.gated_attention = self.gated_attention.to(device)
#         x_g = self.gated_attention(x)
#
#         # self.chunk_attention = self.chunk_attention.to(device)
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         self.v_values = v
#
#         # 原始自注意力
#         attn_ori_return = (q @ k.transpose(-2, -1)) * self.scale
#         attn_ori = attn_ori_return.softmax(dim=-1)
#         attn_ori = self.attn_drop(attn_ori)
#
#         x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
#         x_ori_g = self.gated_attention(x_ori)
#
#         x_ori = self.proj_drop(self.proj(x_ori))
#
#         # GEM
#         xs1 = v
#         xs2 = k
#         xs3 = q
#
#         if self.ss_attn_temp is None:
#             pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
#             inv_temp = pre_norm * self.scale
#         else:
#             inv_temp = self.ss_attn_temp
#
#         for it in range(self.ss_attn_iter):
#             xs1 = F.normalize(xs1, dim=-1)
#             xs2 = F.normalize(xs2, dim=-1)
#             xs3 = F.normalize(xs3, dim=-1)
#
#             attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
#             attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
#             attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp
#
#             attn1 = (attn_return1).softmax(dim=-1)
#             attn2 = (attn_return2).softmax(dim=-1)
#             attn3 = (attn_return3).softmax(dim=-1)
#
#             xs1 = attn1 @ xs1
#             xs2 = attn2 @ xs2
#             xs3 = attn3 @ xs3
#
#         # 最终计算
#         xs1 = F.normalize(xs1, dim=-1)
#         xs2 = F.normalize(xs2, dim=-1)
#         xs3 = F.normalize(xs3, dim=-1)
#
#         attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
#         attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
#         attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp
#
#         attn1 = (attn_return1).softmax(dim=-1)
#         attn2 = (attn_return2).softmax(dim=-1)
#         attn3 = (attn_return3).softmax(dim=-1)
#
#         xs1 = attn1 @ v
#         xs2 = attn2 @ v
#         xs3 = attn3 @ v
#         xs = (xs1 + xs2 + xs3) / 3
#
#         x = xs.transpose(1, 2).reshape(B, N, C)
#
#
#         x = self.proj_drop(self.proj(x))
#
#         # 通过动态路由和分层注意力进一步处理特征
#         # x_g = self.gated_attention(x)
#         # x_ori_g = self.gated_attention(x_ori)
#
#         x =  0.1 * x_g + 0.9 * x
#         x_ori =  0.1 * x_ori_g + 0.9 * x_ori
#         # x = self.chunk_attention(x)
#
#         # # 通过MLP进一步处理特征
#         x_m = self.mlp(x)
#         x_ori_m = self.mlp(x_ori)
#
#         x = 0.9 * x + 0.1 * x_m
#         x_ori = 0.9 * x_ori + 0.1 * x_ori_m
#
#         return [x.transpose(0, 1), x_ori.transpose(0, 1)]

# 修改5




class GEMResidualBlock(nn.Module):
    def __init__(self, res_block):
        super(GEMResidualBlock, self).__init__()
        self.res_block = res_block

    def forward(self,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ):
        if isinstance(q_x, list):
            x_gem, q_x = q_x
        else:
            x_gem = q_x

        x_gem_res, x_ori_res = self.res_block.attn(x=self.res_block.ln_1(q_x))
        x_gem_res, x_ori_res = self.res_block.ls_1(x_gem_res), self.res_block.ls_1(x_ori_res)
        # Original
        x_ori = q_x + x_ori_res
        x_ori = x_ori + self.res_block.ls_2(self.res_block.mlp(self.res_block.ln_2(x_ori)))
        # GEM
        x_gem = x_gem + x_gem_res
        return [x_gem, x_ori]



class EnhancedGEMResidualBlock(nn.Module):
    def __init__(self, res_block):
        super(EnhancedGEMResidualBlock, self).__init__()
        self.res_block = res_block
        self.dim = 896
        self.num_heads = 8

        # 添加注意力机制
        self.self_attn = SelfAttention(dim=896, num_heads=8, dropout=0.)

        # 添加特征融合
        self.feat_fusion = nn.Linear(896 * 2, 896)

        # 添加正则化
        self.drop_path = DropPath(drop_prob=0.1)

    def forward(self,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ):
        if isinstance(q_x, list):
            x_gem, q_x = q_x
        else:
            x_gem = q_x

        # 确保所有张量都在同一个设备上
        device = q_x.device
        self.self_attn.to(device)
        self.feat_fusion.to(device)
        self.drop_path.to(device)

        # 原始残差块
        x_gem_res, x_ori_res = self.res_block.attn(x=self.res_block.ln_1(q_x))
        x_gem_res, x_ori_res = self.res_block.ls_1(x_gem_res), self.res_block.ls_1(x_ori_res)

        # 增强的注意力机制
        x_attn = self.self_attn(q_x)

        # 特征融合
        x_fused = torch.cat([x_gem_res, x_attn], dim=-1)
        x_fused = self.feat_fusion(x_fused)

        # 原始路径
        x_ori = q_x + x_ori_res
        x_ori = x_ori + self.res_block.ls_2(self.res_block.mlp(self.res_block.ln_2(x_ori)))

        # GEM路径
        x_gem = x_gem + x_fused
        x_gem = x_gem + self.drop_path(self.res_block.ls_2(self.res_block.mlp(self.res_block.ln_2(x_gem))))

        return [x_gem, x_ori]



class GEMViT(nn.Module):
    def __init__(self, vit):
        self.vit = vit



# 原始的
def modified_vit_forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid_h, grid_w = x.shape[2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[1]:
            pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
                                             new_size=[grid_h, grid_w],
                                             # old_size=list(self.grid_size),
                                             num_prefix_tokens=1,
                                             interpolation='bicubic',
                                             antialias=True)

        else:
            pos_emb = self.positional_embedding

        x = x + pos_emb.to(x.dtype)
        # x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x_gem, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_gem = x_gem.permute(1, 0, 2)  # LND -> NLD

        # Apply proj
        x = self.ln_post(x)
        x_gem = self.ln_post(x_gem)
        if self.proj is not None:
            x = x @ self.proj
            x_gem = x_gem @ self.proj

        return [x_gem, x]


class MultiScaleAdapter(nn.Module):
    def __init__(self, in_dim):
        super(MultiScaleAdapter, self).__init__()

        # 确保所有分支的输出维度都是 in_dim
        self.branch1 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2, bias=False),  # 减少维度
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加 Dropout
            nn.Linear(in_dim // 2, in_dim, bias=False)  # 恢复维度
        )

        self.branch2 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2, bias=False),  # 减少维度
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加 Dropout
            nn.Linear(in_dim // 2, in_dim, bias=False)  # 恢复维度
        )

        self.branch3 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2, bias=False),  # 减少维度
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),  # 添加 Dropout
            nn.Linear(in_dim // 2, in_dim, bias=False)  # 恢复维度
        )

        # 融合层保持输入输出维度一致
        self.fusion = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim, bias=False),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(0.1)  # 添加 Dropout
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # 残差连接
        b1 += x
        b2 += x
        b3 += x

        return self.fusion(torch.cat([b1, b2, b3], dim=-1))


# 部分x x_gemgenggai
# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)
#     x = x.permute(0, 2, 1)
#
#     # 添加特殊token（CLS token）
#     cls_token = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)
#     x = torch.cat([cls_token, x], dim=1)
#
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
#                                          new_size=[grid_h, grid_w],
#                                          num_prefix_tokens=1,
#                                          interpolation='bicubic',
#                                          antialias=True)
#     else:
#         pos_emb = self.positional_embedding
#
#     x = x + pos_emb.to(x.dtype)
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#
#     x = x.permute(1, 0, 2)  # [seq_len, batch_size, dim]
#
#     # 初始化适配器和融合权重（如果还没有初始化）
#     if not hasattr(self, 'adapters'):
#         self.adapters = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(x.shape[-1], 768, bias=False),
#                 nn.LayerNorm(768),
#                 nn.GELU(),
#                 nn.Linear(768, x.shape[-1], bias=False)
#             ) for _ in range(len(self.transformer.resblocks))
#         ])
#         self.adapters = self.adapters.to(device)
#
#         # 初始化融合权重层
#         self.fusion_weights = nn.Sequential(
#             nn.Linear(640, 640, bias=False),  # 使用proj后的维度
#             nn.LayerNorm(640),
#             nn.GELU(),
#             nn.Linear(640, 1, bias=False),
#             nn.Sigmoid()
#         ).to(device)
#
#     # 通过 transformer 处理特征，并在每一层插入适配器
#     x_gem, x = self.transformer(x)
#     adapter_layers = {5, 11, 17, 23}
#     # 在每一层后应用适配器
#     # for i, adapter in enumerate(self.adapters):
#     #     # 获取当前层的输出
#     #     if adapter is not None and i in adapter_layers:
#     #         current_x = x[i]  # [batch_size, dim]
#     #         # 应用适配器
#     #         adapter_out = adapter(current_x)
#     #
#     #         # 残差连接
#     #         x[i] = 0.8 * current_x + 0.2 * adapter_out
#
#     x = x.permute(1, 0, 2)  # [batch_size, seq_len, dim]
#     x_gem = x_gem.permute(1, 0, 2)
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     # 提取CLS token作为全局表示
#     cls_representation = x[:, 0, :]  # 获取CLS token的表示
#
#     # 将CLS token的表示与原输出融合
#     fusion_weights = self.fusion_weights(cls_representation.unsqueeze(1))
#     x = x * fusion_weights
#     x_gem = x_gem * (1 - fusion_weights)
#
#     # 返回与原函数相同的输出格式
#     return [x_gem, x]


# adapter
# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)
#     x = x.permute(0, 2, 1)
#
#     x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
#
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
#                                          new_size=[grid_h, grid_w],
#                                          num_prefix_tokens=1,
#                                          interpolation='bicubic',
#                                          antialias=True)
#     else:
#         pos_emb = self.positional_embedding
#
#     x = x + pos_emb.to(x.dtype)
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#
#     x = x.permute(1, 0, 2)  # [seq_len, batch_size, dim]
#
#     # 初始化适配器和融合权重（如果还没有初始化）
#     if not hasattr(self, 'adapters'):
#         self.adapters = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(x.shape[-1], 768, bias=False),
#                 nn.LayerNorm(768),
#                 nn.GELU(),
#                 nn.Linear(768, x.shape[-1], bias=False)
#             ) for _ in range(len(self.transformer.resblocks))
#         ])
#         self.adapters = self.adapters.to(device)
#
#         # 初始化融合权重层
#         self.fusion_weights = nn.Sequential(
#             nn.Linear(640, 640, bias=False),  # 使用proj后的维度
#             nn.LayerNorm(640),
#             nn.GELU(),
#             nn.Linear(640, 1, bias=False),
#             nn.Sigmoid()
#         ).to(device)
#
#     # 通过 transformer 处理特征，并在每一层插入适配器
#     x_gem, x = self.transformer(x)
#     adapter_layers = {5, 11, 17, 23}
#     # 在每一层后应用适配器
#     for i, adapter in enumerate(self.adapters):
#         # 获取当前层的输出
#         if adapter is not None and i in adapter_layers:
#             current_x = x[i]  # [batch_size, dim]
#             # 应用适配器
#             adapter_out = adapter(current_x)
#
#             # 残差连接
#             x[i] = 0.8 * current_x + 0.2 * adapter_out
#
#     x = x.permute(1, 0, 2)  # [batch_size, seq_len, dim]
#     x_gem = x_gem.permute(1, 0, 2)
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     # 特征融合
#     # fusion_weights = self.fusion_weights(x)
#     # fusion_weights = 0.5
#     # x = x * fusion_weights
#     # x_gem = x_gem * (1 - fusion_weights)
#
#     return [x_gem, x]

# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [batch_size, 896, 15, 15]
#
#
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [batch_size, 896, 225]
#
#
#     x = x.permute(0, 2, 1)  # shape: [batch_size, 225, 896]
#
#
#     # 添加 class embeddings
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [batch_size, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [batch_size, 226, 896]
#
#
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
#                                          new_size=[grid_h, grid_w],
#                                          num_prefix_tokens=1,
#                                          interpolation='bicubic',
#                                          antialias=True)
#     else:
#         pos_emb = self.positional_embedding
#
#     x = x + pos_emb.to(x.dtype)  # shape: [batch_size, 226, 896]
#     # print(f"After positional embedding shape: {x.shape}")
#
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#
#     x = x.permute(1, 0, 2)  # shape: [226, batch_size, 896]
#     # print(f"After ln_pre and permute shape: {x.shape}")
#
#     # 初始化增强的模块
#     if not hasattr(self, 'enhanced_modules'):
#         self.enhanced_modules = nn.ModuleList([
#             nn.Sequential(
#                 MultiScaleAdapter(x.shape[-1]),
#                 FeatureEnhancement(x.shape[-1]),
#                 DualAttentionModule(x.shape[-1]),
#                 # SpatialAttention(),
#             ) for _ in range(len(self.transformer.resblocks))
#         ])
#         self.enhanced_modules = self.enhanced_modules.to(device)
#
#         # 修改融合权重层的维度以匹配proj后的维度
#         self.fusion_weights = nn.Sequential(
#             nn.Linear(640, 640, bias=False),  # 修改为640以匹配proj后的维度
#             nn.LayerNorm(640),
#             nn.GELU(),
#             nn.Linear(640, 1, bias=False),
#             nn.Sigmoid()
#         ).to(device)
#
#     # 通过 transformer 处理特征
#     x_gem, x = self.transformer(x)  # x的shape: [226, batch_size, 896]
#
#     adapter_layers = {5, 11, 17, 23}
#     # 应用增强的特征提取
#     for i, module in enumerate(self.enhanced_modules):
#         if module is not None and i in adapter_layers:
#             x = x.to(device)
#             # print(f"Before multi_scale shape: {x.shape}")
#             multi_scale = module[0](x)
#             # print(f"After multi_scale shape: {multi_scale.shape}")
#             enhanced = module[1](multi_scale)
#             # print(f"After enhanced shape: {enhanced.shape}")
#             attention = module[2](enhanced)
#             # print(f"After spatical shape: {spatical.shape}")
#             x = x + 0.1 * attention   # 残差连接
#             # print(f"After residual connection shape: {x.shape}")
#     x = x.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     # 特征融合
#     fusion_weights = 0.5
#     x = x * fusion_weights
#     x_gem = x_gem * (1 - fusion_weights)
#
#
#
#     return [x_gem, x]



# 在 modified_vit_forward 中使用 CBAM
# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [1, 896, 15, 15]
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [1, 896, 225]
#     x = x.permute(0, 2, 1)  # shape: [1, 225, 896]
#
#     # 添加 class embeddings
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [1, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [1, 226, 896]
#
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
#                                          new_size=[grid_h, grid_w],
#                                          num_prefix_tokens=1,
#                                          interpolation='bicubic',
#                                          antialias=True)
#     else:
#         pos_emb = self.positional_embedding
#
#     x = x + pos_emb.to(x.dtype)  # shape: [1, 226, 896]
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#     x = x.permute(1, 0, 2)  # shape: [226, 1, 896]
#
#     # 初始化增强的模块
#     adapter_layers = {5, 11, 17, 23}  # 指定插入适配器的层
#     # 初始化增强的模块
#     if not hasattr(self, 'enhanced_modules'):
#         self.enhanced_modules = nn.ModuleList([
#             nn.Sequential(
#                 CBAM(896),  # 使用 CBAM 进行特征增强
#             ) if i in adapter_layers else nn.Identity()  # 其他层使用 Identity
#             for i in range(len(self.transformer.resblocks))
#         ])
#         self.enhanced_modules = self.enhanced_modules.to(device)
#
#     # 通过 transformer 处理特征
#     x_gem, x = self.transformer(x)  # x的shape: [226, 1, 896]
#     # 应用增强的特征提取
#
#     for i, module in enumerate(self.enhanced_modules):
#         x = x.to(device)
#         print( f"Before multi_scale shape: {x.shape}")
#         if i in adapter_layers:
#             multi_scale = module(x)  # 仅在指定层使用 CBAM
#             multi_scale = multi_scale + x  # 残差连接
#
#             x = 0.9 * x + 0.1 * multi_scale  # 残差连接
#             print( f"After multi_scale shape: {x.shape}")
#
#
#     # if not hasattr(self, 'enhanced_modules'):
#     #     self.enhanced_modules = nn.ModuleList([
#     #         nn.Sequential(
#     #
#     #             CBAM(896),  # 使用 CBAM 进行特征增强
#     #             # MultiScaleAdapter(896),  # 输入通道数为896
#     #             # FeatureEnhancement(896),  # 输入通道数为896
#     #
#     #         ) for _ in range(len(self.transformer.resblocks))
#     #     ])
#     #     self.enhanced_modules = self.enhanced_modules.to(device)
#     #
#     # # 通过 transformer 处理特征
#     # x_gem, x = self.transformer(x)  # x的shape: [226, 1, 896]
#     #
#     # # 应用增强的特征提取
#     # adapter_layers = {5, 11, 17, 23}  # 指定插入适配器的层
#     # for i, module in enumerate(self.enhanced_modules):
#     #     if i in adapter_layers:
#     #         x = x.to(device)
#     #         multi_scale = module[0](x)  # MultiScaleAdapter
#     #         multi_scale = multi_scale + x
#     #
#     #         # enhanced = module[1](multi_scale)  # FeatureEnhancement
#     #         # enhanced = enhanced + multi_scale
#     #         #
#     #         # channel = module[2](enhanced)  # CBAM
#     #         # channel = channel + enhanced
#     #
#     #         x = 0.9 * x + 0.1 * multi_scale  # 残差连接
#
#     x = x.permute(1, 0, 2)  # shape: [1, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [1, 226, 896]
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     return [x_gem, x]


# 24层分开
# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [1, 896, 15, 15]
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [1, 896, 225]
#     x = x.permute(0, 2, 1)  # shape: [1, 225, 896]
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [1, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [1, 226, 896]
#
#     # 处理位置编码
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(
#             self.positional_embedding.unsqueeze(0),
#             new_size=[grid_h, grid_w],
#             num_prefix_tokens=1,
#             interpolation='bicubic',
#             antialias=True
#         )
#     else:
#         pos_emb = self.positional_embedding
#     x = x + pos_emb.to(x.dtype)
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#     x = x.permute(1, 0, 2)  # shape: [226, 1, 896]
#
#     adapter_layers = {5, 11, 17, 23}  # 假设这是你指定的层索引（从0开始）
#
#     # 逐层处理 Transformer，并插入增强模块
#     x_all = []  # 存储所有层的输出（用于生成 x_gem）
#     for i, resblock in enumerate(self.transformer.resblocks):
#         # print( f"Before multi_scale shape: {x}")
#         x = resblock(x)  # 通过当前的 Transformer 层
#         if isinstance(x, list):
#             x = x[1]
#
#         # print( f"After multi_scale shape: {x}")
#         # 应用增强模块（在当前层之后）
#         if i in adapter_layers:
#             cbam = CBAM(896).to(device)  # 仅在适配器层应用增强模块，传递通道数
#             cbam = cbam(x)  # 使用 x 作为输入
#             x = 0.9 * x + 0.1 * cbam  # 残差连接
#         # print( f"After multi_scale shape: {x}")
#         # 保存当前层的输出（用于后续生成 x_gem）
#         x_all.append(x.detach().clone())  # 使用 detach() 确保不跟踪梯度
#
#     # 生成 x_gem（通过对所有层的输出进行平均）
#     x_gem = torch.mean(torch.stack(x_all), dim=0)  # 对所有层的输出取平均
#
#     # 恢复维度顺序
#     x = x.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     return [x_gem, x]  # 返回 x_gem 和最终输出 x

# 24 two
# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [1, 896, 15, 15]
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [1, 896, 225]
#     x = x.permute(0, 2, 1)  # shape: [1, 225, 896]
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [1, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [1, 226, 896]
#
#     # 处理位置编码
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(
#             self.positional_embedding.unsqueeze(0),
#             new_size=[grid_h, grid_w],
#             num_prefix_tokens=1,
#             interpolation='bicubic',
#             antialias=True
#         )
#     else:
#         pos_emb = self.positional_embedding
#     x = x + pos_emb.to(x.dtype)
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#     x = x.permute(1, 0, 2)  # shape: [226, 1, 896]
#     if not hasattr(self, 'adapters'):
#         self.adapters = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(x.shape[-1], 768, bias=False),
#                 nn.LayerNorm(768),
#                 nn.GELU(),
#                 nn.Linear(768, x.shape[-1], bias=False)
#             ) for _ in range(len(self.transformer.resblocks))
#         ])
#
#     adapter_layers = {2, 5, 8, 11}  # 指定适配器层索引
#     ca  = CategorySpecificAdapter(input_dim=896, num_categories=15).to(device)
#     # 逐层处理 Transformer，并插入增强模块
#     x_all = []  # 存储所有层的输出（用于生成 x_gem）
#
#     # for i in range(24):
#     #     if (i + 1) in adapter_layers:
#     #         x = self.transformer.resblocks[i](x)
#
#     for i, resblock in enumerate(self.transformer.resblocks):
#         # print( f"Before multi_scale shape: {i}")
#         # 通过当前的 Transformer 层
#         x = resblock(x)
#         if isinstance(x, list):
#             # 方法 1：取最后一个元素
#             x = x[1]
#             # 方法 2：将所有元素沿某个维度拼接
#             # x = torch.cat(x, dim=1)
#             # 方法 3：将所有元素取平均
#             # x_mean = torch.mean(torch.stack(x), dim=0)
#             # # 生成 x_gem（通过对所有层的输出进行最大池化）
#             # x_max = torch.max(torch.stack(x), dim=0)[0]
#             # x = x_mean + x_max
#
#             # x_tensor = torch.stack(x)
#             # activated_x = F.leaky_relu(x_tensor, negative_slope=0.01)  # 使用 LeakyReLU
#             # x = torch.mean(activated_x, dim=0)
#             # 应用增强模块（在当前层之后）
#         if i in adapter_layers:
#             # print( f"after adapter_layers shape: {i}")
#             # adapter_out = adapters(x)
#
#             # cbam = CBAM(896).to(device)  # 初始化 CBAM 模块
#             # multi = MultiScaleAdapter(896).to(device) # 输入通道数为896
#             # multi_out = multi(x)
#             # cbam_output = cbam(x)  # 应用 CBAM
#             category = ca(x)
#             x = 0.9 * x + 0.1 * category
#             # 残差连接
#
#         # 保存当前层的输出（用于后续生成 x_gem）
#         x_all.append(x.detach().clone())
#
#     x_gem_1, x_1 = self.transformer(x)
#     x = x + x_1
#     # 生成 x_gem（通过对所有层的输出进行平均）
#     x_gem = torch.mean(torch.stack(x_all), dim=0)
#     x_gem = x_gem_1 + x_gem
#     # # 生成 x_gem（通过对所有层的输出进行最大池化）
#     # x_gem_max = torch.max(torch.stack(x_all), dim=0)[0]
#     # x_gem = x_gem_mean + x_gem_max
#     # 恢复维度顺序
#     x = x.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     return [x_gem, x]

# 修改
# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [1, 896, 15, 15]
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [1, 896, 225]
#     x = x.permute(0, 2, 1)  # shape: [1, 225, 896]
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [1, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [1, 226, 896]
#
#     # 处理位置编码
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(
#             self.positional_embedding.unsqueeze(0),
#             new_size=[grid_h, grid_w],
#             num_prefix_tokens=1,
#             interpolation='bicubic',
#             antialias=True
#         )
#     else:
#         pos_emb = self.positional_embedding
#     x = x + pos_emb.to(x.dtype)
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#     x = x.permute(1, 0, 2)  # shape: [226, 1, 896]
#
#     # if not hasattr(self, 'adapters'):
#     #     self.adapters = nn.ModuleList([
#     #         nn.Sequential(
#     #             nn.Linear(x.shape[-1], 768, bias=False),
#     #             nn.LayerNorm(768),
#     #             nn.GELU(),
#     #             nn.Linear(768, x.shape[-1], bias=False)
#     #         ) for _ in range(len(self.transformer.resblocks))
#     #     ]).to(device)
#
#     adapter_layers = {2, 5, 7, 8, 9, 11}  # 指定适配器层索引
#     ca = CategorySpecificAdapter(input_dim=896, num_categories=10).to(device)
#
#     # 逐层处理 Transformer，并插入增强模块
#     x_all = []  # 存储所有层的输出（用于生成 x_gem）
#
#     for i, resblock in enumerate(self.transformer.resblocks):
#         # 通过当前的 Transformer 层
#         iden = x
#         x = resblock(x)
#         if isinstance(x, list):
#             x = x[1]
#
#         if i in adapter_layers:
#             # 应用类别特定适配器
#             category = ca(x)
#             # 使用可学习的权重进行融合
#             x = iden + 0.9 * x + 0.1 * category
#
#         # 保存当前层的输出（用于后续生成 x_gem）
#         x_all.append(x.detach().clone())
#
#     # 生成 x_gem（通过对所有层的输出进行加权求和）
#     x_all_stack = torch.stack(x_all)
#     # 使用注意力机制动态调整权重
#     attn_weights = torch.softmax(torch.mean(x_all_stack, dim=-1), dim=0)
#     x_gem = torch.sum(attn_weights.unsqueeze(-1) * x_all_stack, dim=0)
#     x_gem_1, x_1 = self.transformer(x)
#
#     x = 0.8 * x + 0.5 * x_1
#     x_gem = 0.8 * x_gem  + 0.5 * x_gem_1
#     # 恢复维度顺序
#     x = x.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     return [x_gem, x]

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # 缩放因子

        # 定义查询（Q）、键（K）和值（V）的线性变换
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape: [batch_size, seq_length, dim]
        batch_size, seq_length, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_length, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 分离 Q, K, V

        # 计算注意力权重
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_length, dim)
        output = self.proj(output)
        output = self.dropout(output)

        return output


def adaptive_normalize(x: torch.Tensor, eps=1e-6):
    # x.shape: [batch_size, seq_length, dim]
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x_norm = (x - mean) / (std + eps)

    # 引入可学习的缩放和偏移参数
    gamma = nn.Parameter(torch.ones(x.shape[-1])).to(x.device)
    beta = nn.Parameter(torch.zeros(x.shape[-1])).to(x.device)

    return gamma * x_norm + beta



# 第二次修改
# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [1, 896, 15, 15]
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [1, 896, 225]
#     x = x.permute(0, 2, 1)  # shape: [1, 225, 896]
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [1, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [1, 226, 896]
#
#     # 处理位置编码
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(
#             self.positional_embedding.unsqueeze(0),
#             new_size=[grid_h, grid_w],
#             num_prefix_tokens=1,
#             interpolation='bicubic',
#             antialias=True
#         )
#     else:
#         pos_emb = self.positional_embedding
#     x = x + pos_emb.to(x.dtype)
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#     x = x.permute(1, 0, 2)  # shape: [226, 1, 896]
#
#     x_gem_1, x_1 = self.transformer(x)
#     # 引入多尺度特征融合
#     multiscale_features = []
#     adapter_layers = {2, 5, 8, 11}  # 指定适配器层索引
#     ca = CategorySpecificAdapter(input_dim=896, num_categories=10).to(device)
#
#     # 逐层处理 Transformer，并插入增强模块
#     x_all = []  # 存储所有层的输出（用于生成 x_gem）
#
#     for i, resblock in enumerate(self.transformer.resblocks):
#         # 通过当前的 Transformer 层
#         iden = x
#         x = resblock(x)
#         if isinstance(x, list):
#             x = x[1]
#
#         if i in adapter_layers:
#             # 应用类别特定适配器
#             category = ca(x)
#             # 使用可学习的权重进行融合
#             x = iden + 0.9 * x + 0.1 * category
#
#         # 保存当前层的输出（用于后续生成 x_gem）
#         x_all.append(x.detach().clone())
#
#         # 多尺度特征融合
#         if i in {2, 5, 8}:  # 在特定层保存多尺度特征
#             multiscale_features.append(x.detach().clone())
#
#     # 生成 x_gem（通过对所有层的输出进行加权求和）
#     x_all_stack = torch.stack(x_all)
#     # 使用注意力机制动态调整权重
#     attn_weights = torch.softmax(torch.mean(x_all_stack, dim=-1), dim=0)
#     x_gem = torch.sum(attn_weights.unsqueeze(-1) * x_all_stack, dim=0)
#
#
#     # 多尺度特征融合
#     multiscale_weights = torch.softmax(torch.tensor([0.2, 0.3, 0.5]), dim=0).to(device)
#     multiscale_fused = sum(w * f for w, f in zip(multiscale_weights, multiscale_features))
#     # 特征融合
#     x = 0.3 * x + 0.6 * x_1 + 0.1 * multiscale_fused
#     x_gem = 0.3 * x_gem + 0.6 * x_gem_1 + 0.1 * multiscale_fused
#     # 恢复维度顺序
#     x = x.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#     # multiscale_fused = multiscale_fused.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#
#
#
#     # # 引入自注意力机制增强特征表达
#     # sa = CBAM(896).to(device)
#     # x = sa(x)
#     # x_gem = sa(x_gem)
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     # 自适应归一化
#     x = adaptive_normalize(x)
#     x_gem = adaptive_normalize(x_gem)
#
#     return [x_gem, x]


# 第三次修改
# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [1, 896, 15, 15]
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [1, 896, 225]
#     x = x.permute(0, 2, 1)  # shape: [1, 225, 896]
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [1, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [1, 226, 896]
#
#     # 处理位置编码
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(
#             self.positional_embedding.unsqueeze(0),
#             new_size=[grid_h, grid_w],
#             num_prefix_tokens=1,
#             interpolation='bicubic',
#             antialias=True
#         )
#     else:
#         pos_emb = self.positional_embedding
#     x = x + pos_emb.to(x.dtype)
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#     x = x.permute(1, 0, 2)  # shape: [226, 1, 896]
#
#     x_gem_1, x_1 = self.transformer(x)
#     # 引入多尺度特征融合
#     multiscale_features = []
#     adapter_layers = {2, 5, 8, 11}  # 指定适配器层索引
#     ca = CategorySpecificAdapter(input_dim=896, num_categories=10).to(device)
#     self.ln_adapter = nn.LayerNorm(896).to(device)  # 添加归一化层
#
#     # 逐层处理 Transformer，并插入增强模块
#     x_all = []  # 存储所有层的输出（用于生成 x_gem）
#
#     for i, resblock in enumerate(self.transformer.resblocks):
#         # 通过当前的 Transformer 层
#         iden = x
#         x = resblock(x)
#         if isinstance(x, list):
#             x = x[1]
#
#         if i in adapter_layers:
#             # 应用类别特定适配器
#             x_norm = self.ln_adapter(x)  # 添加归一化层
#             category = ca(x_norm)
#             x = iden + 0.9 * x + 0.1 * category
#
#         # 保存当前层的输出（用于后续生成 x_gem）
#         x_all.append(x.detach().clone())
#
#         # 多尺度特征融合
#         if i in {2, 5, 8}:  # 在特定层保存多尺度特征
#             multiscale_features.append(x.detach().clone())
#
#     # 生成 x_gem（通过对所有层的输出进行加权求和）
#     x_all_stack = torch.stack(x_all)
#     attn_weights = torch.softmax(torch.mean(x_all_stack, dim=-1), dim=0)
#     x_gem = torch.sum(attn_weights.unsqueeze(-1) * x_all_stack, dim=0)
#
#     # 多尺度特征融合
#     multiscale_weights = nn.Parameter(torch.tensor([0.5, 0.5, 0.5])).to(device)
#     multiscale_weights = torch.softmax(multiscale_weights, dim=0)
#     multiscale_fused = sum(w * f for w, f in zip(multiscale_weights, multiscale_features))
#
#     # 动态调整特征融合权重
#     fusion_weights = torch.softmax(torch.tensor([0.1, 1.0, 0.1]), dim=0).to(device)
#     x = fusion_weights[0] * x + fusion_weights[1] * x_1 + fusion_weights[2] * multiscale_fused
#     x_gem = fusion_weights[0] * x_gem + fusion_weights[1] * x_gem_1 + fusion_weights[2] * multiscale_fused
#
#     # 恢复维度顺序
#     x = x.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#
#     # 引入自注意力机制增强特征表达
#     # sa = SelfAttention(dim=896, num_heads=8, dropout=0.1).to(device)
#     # x = sa(x)
#     # x_gem = sa(x_gem)
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     # 自适应归一化
#     def adaptive_normalize(t):
#         mean = t.mean(dim=-1, keepdim=True)
#         std = t.std(dim=-1, keepdim=True)
#         return (t - mean) / (std + 1e-6)
#
#     x = adaptive_normalize(x)
#     x_gem = adaptive_normalize(x_gem)
#
#     return [x_gem, x]



# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [1, 896, 15, 15]
#     # print("After conv1 shape:", x.shape)
#
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [1, 896, 225]
#     # print("After reshape shape:", x.shape)
#
#     x = x.permute(0, 2, 1)  # shape: [1, 225, 896]
#     # print("After permute shape:", x.shape)
#
#     # 添加 class embeddings
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [1, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [1, 226, 896]
#     # print("After adding class embedding shape:", x.shape)
#
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
#                                          new_size=[grid_h, grid_w],
#                                          num_prefix_tokens=1,
#                                          interpolation='bicubic',
#                                          antialias=True)
#     else:
#         pos_emb = self.positional_embedding
#
#     x = x + pos_emb.to(x.dtype)  # shape: [1, 226, 896]
#     # print("After adding positional embedding shape:", x.shape)
#
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#     # print("After patch dropout and layer norm shape:", x.shape)
#
#     x = x.permute(1, 0, 2)  # shape: [226, 1, 896]
#     # print("After permute for transformer shape:", x.shape)
#
#     # 初始化增强的模块
#     if not hasattr(self, 'enhanced_modules'):
#         self.enhanced_modules = nn.ModuleList([
#             nn.Sequential(
#                 MultiScaleAdapter(896),  # 输入通道数为896
#                 FeatureEnhancement(896),  # 输入通道数为896
#                 # SimpleSpatialAttention(),  # 空间注意力机制
#                 # ChannelAttentionBlock(896),  # 通道注意力机制
#                 CBAM(896),
#                 # ContextAwareAdapter(896),  # 上下文信息计算
#                 # AttentionModule(896)  # 注意力模块
#             ) for _ in range(len(self.transformer.resblocks))
#         ])
#         self.enhanced_modules = self.enhanced_modules.to(device)
#
#         # 初始化融合权重层
#         # self.fusion_weights = nn.Sequential(
#         #     nn.Linear(896, 896, bias=False),  # 输入通道数为896
#         #     nn.LayerNorm(896),
#         #     nn.GELU(),
#         #     nn.Linear(896, 1, bias=False),
#         #     nn.Sigmoid()
#         # ).to(device)
#
#     # 通过 transformer 处理特征
#     x_gem, x = self.transformer(x)  # x的shape: [226, 1, 896]
#     # print("After transformer output shape:", x.shape)
#
#     # 应用增强的特征提取
#     adapter_layers = {5, 11, 17, 23}  # 指定插入适配器的层
#     for i, module in enumerate(self.enhanced_modules):
#         if i in adapter_layers:
#             x = x.to(device)
#             # print(f"Before module {i} shape:", x.shape)
#             multi_scale = module[0](x)  # MultiScaleAdapter
#             # print(f"After MultiScaleAdapter {i} shape:", multi_scale.shape)
#             enhanced = module[1](multi_scale)  # FeatureEnhancement
#             # print(f"After FeatureEnhancement {i} shape:", enhanced.shape)
#
#             channel = module[2](enhanced)  # ChannelAttentionBlock
#             # print(f"After ContextAwareAdapter {i} shape:", channel.shape)
#             # attention = module[3](channel)  # AttentionModule
#             # print(f"After AttentionModule {i} shape:", attention.shape)
#             x = 0.9 * x + 0.1 * channel  # 残差连接
#             # print(f"After residual connection {i} shape:", x.shape)
#
#     x = x.permute(1, 0, 2)  # shape: [1, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [1, 226, 896]
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     # 特征融合
#     # fusion_weights = self.fusion_weights(x)  # 使用融合权重
#     # x = x * fusion_weights
#     # x_gem = x_gem * (1 - fusion_weights)
#
#     # print("Final output shape:", x.shape)
#     return [x_gem, x]



# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [1, 896, 15, 15]
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [1, 896, 225]
#     x = x.permute(0, 2, 1)  # shape: [1, 225, 896]
#
#     # 添加 class embeddings
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [1, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [1, 226, 896]
#
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
#                                          new_size=[grid_h, grid_w],
#                                          num_prefix_tokens=1,
#                                          interpolation='bicubic',
#                                          antialias=True)
#     else:
#         pos_emb = self.positional_embedding
#
#     x = x + pos_emb.to(x.dtype)  # shape: [1, 226, 896]
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#
#     x = x.permute(1, 0, 2)  # shape: [226, 1, 896]
#
#     # 初始化增强的模块
#     if not hasattr(self, 'enhanced_modules'):
#         self.enhanced_modules = nn.ModuleList([
#             nn.Sequential(
#                 MultiScaleAdapter(896),  # 输入通道数为896
#                 ContextAwareAdapter(896),
#                 FeatureEnhancement(896),  # 输入通道数为896
#                 # SimpleSpatialAttention(),  # 空间注意力机制
#                 # ChannelAttentionBlock(896),  # 通道注意力机制
#                 CBAM(896),  # CBAM模块
#             ) for _ in range(len(self.transformer.resblocks))
#         ])
#         self.enhanced_modules = self.enhanced_modules.to(device)
#
#         # 初始化融合权重层
#         # self.fusion_weights = nn.Sequential(
#         #     nn.Linear(896, 896, bias=False),  # 输入通道数为896
#         #     nn.LayerNorm(896),
#         #     nn.GELU(),
#         #     nn.Linear(896, 1, bias=False),
#         #     nn.Sigmoid()
#         # ).to(device)
#
#     # 通过 transformer 处理特征
#     x_gem, x = self.transformer(x)  # x的shape: [226, 1, 896]
#
#     # 应用增强的特征提取
#     adapter_layers = {5, 11, 17, 23}  # 指定插入适配器的层
#     for i, module in enumerate(self.enhanced_modules):
#         if i in adapter_layers:
#             x = x.to(device)
#             # print(f"Before module {i} shape:", x.shape)
#             enhanced = module[0](x)  # MultiScaleAdapter
#             # print(f"After MultiScaleAdapter {i} shape:", enhanced.shape)
#             context = module[1](enhanced)
#             enhanced = module[2](context)  # FeatureEnhancement
#             cbam = module[3](enhanced)
#             # spatial = module[3](enhanced)
#             # channel = module[4](spatial)
#             # print(f"After FeatureEnhancement {i} shape:", enhanced.shape)
#             # attention = module[2](enhanced)  # CBAM
#             # print(f"After CBAM {i} shape:", attention.shape)
#             x = 0.9 * x + 0.1 * cbam  # 残差连接
#             # print(f"After residual connection {i} shape:", x.shape)
#
#     x = x.permute(1, 0, 2)  # shape: [1, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [1, 226, 896]
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     # 特征融合
#     # fusion_weights = self.fusion_weights(x)  # 使用融合权重
#     # x = x * fusion_weights
#     # x_gem = x_gem * (1 - fusion_weights)
#
#     return [x_gem, x]


# def modified_vit_forward(self, x: torch.Tensor):
#     device = x.device
#
#     # 初始处理保持不变
#     x = self.conv1(x)  # shape: [batch_size, 896, 15, 15]
#     grid_h, grid_w = x.shape[2:]
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape: [batch_size, 896, 225]
#     x = x.permute(0, 2, 1)  # shape: [batch_size, 225, 896]
#
#     # 添加 class embeddings
#     class_embedding = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype)  # shape: [batch_size, 1, 896]
#     x = torch.cat([class_embedding, x], dim=1)  # shape: [batch_size, 226, 896]
#
#     if x.shape[1] != self.positional_embedding.shape[1]:
#         pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
#                                          new_size=[grid_h, grid_w],
#                                          num_prefix_tokens=1,
#                                          interpolation='bicubic',
#                                          antialias=True)
#     else:
#         pos_emb = self.positional_embedding
#
#     x = x + pos_emb.to(x.dtype)  # shape: [batch_size, 226, 896]
#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#
#     x = x.permute(1, 0, 2)  # shape: [226, batch_size, 896]
#
#     # 初始化适配器和融合权重（如果还没有初始化）
#     if not hasattr(self, 'adapters'):
#         self.adapters = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(x.shape[-1], 768, bias=False),
#                 nn.LayerNorm(768),
#                 nn.GELU(),
#                 nn.Linear(768, x.shape[-1], bias=False)
#             ) for _ in range(len(self.transformer.resblocks))
#         ])
#         self.adapters = self.adapters.to(device)
#
#         # 初始化融合权重层
#         self.fusion_weights = nn.Sequential(
#             nn.Linear(896, 896, bias=False),  # 使用proj后的维度
#             nn.LayerNorm(896),
#             nn.GELU(),
#             nn.Linear(896, 1, bias=False),
#             nn.Sigmoid()
#         ).to(device)
#
#         # 初始化 CBAM 模块
#         self.cbam = CBAM(896)  # 假设 CBAM 的输入通道数为 896
#         self.cbam = self.cbam.to(device)
#
#     # 通过 transformer 处理特征，并在每一层插入适配器
#     x_gem, x = self.transformer(x)  # x的shape: [226, batch_size, 896]
#
#     adapter_layers = {5, 11, 17, 23}  # 指定插入适配器的层
#     # 在每一层后应用适配器
#     for i, adapter in enumerate(self.adapters):
#         if adapter is not None and i in adapter_layers:
#             current_x = x[i]  # [batch_size, dim]
#             # 应用适配器
#             adapter_out = adapter(current_x)
#
#             # 残差连接
#             x[i] = current_x + adapter_out
#
#     # 应用 CBAM
#     x = self.cbam(x)  # 通过 CBAM 模块增强特征
#     print(f"After CBAM shape: {x.shape}")  # 打印 CBAM 后的形状
#
#     x = x.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#     x_gem = x_gem.permute(1, 0, 2)  # shape: [batch_size, 226, 896]
#
#     # 后处理
#     x = self.ln_post(x)
#     x_gem = self.ln_post(x_gem)
#     if self.proj is not None:
#         x = x @ self.proj
#         x_gem = x_gem @ self.proj
#
#     # 特征融合
#     fusion_weights = self.fusion_weights(x)  # 使用融合权重
#     x = x * fusion_weights
#     x_gem = x_gem * (1 - fusion_weights)
#
#     return [x_gem, x]

