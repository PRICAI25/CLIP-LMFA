import json
import os

import math
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch._inductor.utils import cache_dir
from torch.cuda import device_of

from utils.SCSA_two import EnhancedSimpleCBAM, EnhancedImageProcessor, S2Attention, PSA, ECAAttention

# 示例用法
# gem_model 是一个假设的模型对象，具有 encode_text 方法和 embedding_dim 属性
# text 是输入的文本列表
# text_embeddings = extract_text_embeddings_by_batch(gem_model, text)
import torch
import torch.nn as nn
# 主要
import math
import torch
from sklearn.cluster import KMeans

from utils.test_adapter import train_model, load_model_weights
from utils.test_adapter_three import CLIP_Inplanted


# def extract_text_embeddings_by_batch(gem_model, text, batch_size=64, num_clusters=10):
#     # Step 1: Extract initial text embeddings for clustering
#     initial_embeddings = []
#     num_batches = math.ceil(len(text) / batch_size)
#
#     for i in range(num_batches):
#         batch_text = text[i * batch_size: (i + 1) * batch_size]
#         embeddings = gem_model.encode_text(batch_text).squeeze(0).tolist()
#         initial_embeddings.extend(embeddings)
#
#     # Step 2: Apply K-Means clustering on the initial embeddings
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(initial_embeddings)
#
#     # Step 3: Group text by cluster
#     clustered_texts = {i: [] for i in range(num_clusters)}
#     for idx, label in enumerate(kmeans.labels_):
#         clustered_texts[label].append(text[idx])
#
#     # Step 4: Extract embeddings for each cluster and average them
#     text_embeddings = []
#     for cluster in clustered_texts.values():
#         if cluster:  # Ensure the cluster is not empty
#             cluster_embeddings = gem_model.encode_text(cluster).squeeze(0)
#             avg_embedding = torch.mean(cluster_embeddings, dim=0).tolist()  # Average the embeddings
#             text_embeddings.append(avg_embedding)
#
#     return text_embeddings
def extract_text_embeddings_by_batch(gem_model, text, batch_size=64):
    num_batches = math.ceil(len(text) / batch_size)
    text_embeddings = []
    for i in range(num_batches):
        text_embeddings.extend(
            gem_model.encode_text(text[i * batch_size : (i + 1) * batch_size])
            .squeeze(0)
            .tolist()
        )
    return text_embeddings

# 定义了一个名为 extract_text_embeddings 的函数，用于从给定的 JSON 文件中提取文本嵌入向量，并根据指定的返回类型返回结果。
def extract_text_embeddings(prompt_path, gem_model, classname="", return_type="pandas"):
    # 读取提示文件：从指定路径读取 JSON 文件并加载为 prompts。
    with open(prompt_path) as fp:
        prompts = json.load(fp)

    with torch.no_grad():
        # 格式化正常文本：使用 classname 格式化正常文本。
        normal_text = [
            t.format(classname=classname) for t in prompts["normal"]["prompts"]
        ]
        # 格式化异常文本：使用 classname 格式化异常文本
        abnormal_text = [
            t.format(classname=classname) for t in prompts["abnormal"]["prompts"]
        ]
        # 提取正常文本嵌入向量：调用 extract_text_embeddings_by_batch 提取正常文本的嵌入向量。
        normal_text_embeddings = extract_text_embeddings_by_batch(
            gem_model, normal_text, batch_size=64
        )
        # 提取异常文本嵌入向量：调用 extract_text_embeddings_by_batch 提取异常文本的嵌入向量
        abnormal_text_embeddings = extract_text_embeddings_by_batch(
            gem_model, abnormal_text, batch_size=64
        )
        # 返回 Pandas DataFrame：如果是 "pandas"，返回包含文本、特征向量、特征类型和标签的 Pandas DataFrame
        if return_type == "pandas":
            return pd.DataFrame(
                {
                    "text": normal_text + abnormal_text,
                    "feature": normal_text_embeddings + abnormal_text_embeddings,
                    "feature_type": "text",
                    "feature_label": ["good"] * len(normal_text_embeddings)
                    + ["anomaly"] * len(abnormal_text_embeddings),
                }
            )
        # 如果 return_type 是 "numpy"，则返回两个 NumPy 数组，一个包含所有特征向量，另一个包含对应的标签。
        elif return_type == "numpy":
            X = np.array(normal_text_embeddings + abnormal_text_embeddings)
            y = np.array(
                [0] * len(normal_text_embeddings) + [1] * len(abnormal_text_embeddings)
            )
            return X, y

# 定义了一个函数 extract_all_text_embeddings，用于从多个提示路径中提取文本嵌入向量及其标签。
def extract_all_text_embeddings(prompt_paths, gem_model, classname=""):
    all_text_embeddings, all_text_labels = [], []
    # 遍历每个 prompt_path，调用 extract_text_embeddings 函数提取文本嵌入向量和标签，并将结果分别追加到 all_text_embeddings 和 all_text_labels 列表中。
    for prompt_path in prompt_paths:
        text_embeddings, text_labels = extract_text_embeddings(
            prompt_path, gem_model, classname, return_type="numpy"
        )
        all_text_embeddings.append(text_embeddings)
        all_text_labels.append(text_labels)
    #  axis=0 表示沿着第一个维度（即行方向）进行连接
    all_text_embeddings = np.concatenate(all_text_embeddings, axis=0)
    all_text_labels = np.concatenate(all_text_labels, axis=0)
    return all_text_embeddings, all_text_labels


def extract_image_embeddings(
    multiscale_images: dict[int, torch.Tensor], gem_model, device
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    with torch.no_grad():
        features = {}
        for img_size, images in multiscale_images.items():
            features_gem, features_clip = gem_model.model.visual(images.to(device))

            features_gem = F.normalize(features_gem, dim=-1).detach().cpu().numpy()
            features_clip = F.normalize(features_clip, dim=-1).detach().cpu().numpy()
            features[img_size] = {
                "gem": {
                    "cls": features_gem[:, 0, :],
                    "patch": features_gem[:, 1:, :],
                },
                "clip": {
                    "cls": features_clip[:, 0, :],
                    "patch": features_clip[:, 1:, :],
                },
            }
    return features
def extract_image_embeddings_one(multiscale_images: dict[int, torch.Tensor], gem_model, device) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    with torch.no_grad():
        features = {}
        clip_implanted = CLIP_Inplanted(gem_model, [4, 6, 8, 12], device)  # Assuming 'features' is defined
        for img_size, images in multiscale_images.items():
            features_gem, features_clip = clip_implanted(images.to(device))

            features_gem = features_gem.detach().cpu().numpy()
            features_clip = features_clip.detach().cpu().numpy()
            features[img_size] = {
                "gem": {
                    "cls": features_gem[:, 0, :],
                    "patch": features_gem[:, 1:, :],
                },
                "clip": {
                    "cls": features_clip[:, 0, :],
                    "patch": features_clip[:, 1:, :],
                },
            }
    return features
# Assuming images are RGB with 3 channels
def extract_image_embeddings_add(       # 最高的
        multiscale_images: dict[int, torch.Tensor],
        gem_model,
        device
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    with torch.no_grad():
        features = {}
        block = ECAAttention(kernel_size=3)
        for img_size, images in multiscale_images.items():
            al = 0.49
            images_en =  block(images) + (1 - al) * images
            # images_en =    images
            # 获取原始特征
            features_gem, features_clip = gem_model.model.visual(images.to(device))
            # features_gem.shape 640
            # features_gem.shape 226
            # features_gem.shape 640
            # features_gem.shape 785
            # features_gem.shape 640
            # features_gem.shape 3137
            # print("features_gem.shape",features_gem.shape[-1])
            # print("features_gem.shape",features_gem.shape[1])

            # 创建并应用增强模块
            cbam = EnhancedSimpleCBAM(
                in_channels=3,
                feature_dim=features_gem.shape[-1],
                seq_length=features_gem.shape[1]

            ).to(device)

            # cbam.load_state_dict(torch.load('btad-best_model.pth'))
            # 获取增强特征
            enhanced_gem, enhanced_clip = cbam(images.to(device))

            # # 特征融合
            alpha = 0.9  # 可调节的权重参数
            features_gem =  features_gem + (1 - alpha) * enhanced_gem
            features_clip = features_clip + (1 - alpha) * enhanced_clip
            # #

            # 特征融合，使用可学习的权重
            # alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的权重
            # # print( "alpha: ",alpha)
            # features_gem = alpha * features_gem + (1 - alpha) * enhanced_gem
            # features_clip = alpha * features_clip + (1 - alpha) * enhanced_clip
            # alpha_1 = 0.985
            # features_gem = alpha_1 * features_gem + (1 - alpha_1) * enhanced_gem
            # features_clip = alpha_1 * features_clip + (1 - alpha_1) * enhanced_clip
            # 标准化
            features_gem = F.normalize(features_gem, dim=-1).detach().cpu().numpy()
            features_clip = F.normalize(features_clip, dim=-1).detach().cpu().numpy()

            features[img_size] = {
                "gem": {"cls": features_gem[:, 0, :], "patch": features_gem[:, 1:, :]},
                "clip": {"cls": features_clip[:, 0, :], "patch": features_clip[:, 1:, :]}
            }
    return features
# def extract_image_embeddings(
#         multiscale_images: dict[int, torch.Tensor],
#         gem_model,
#         device
# ) -> dict[int, dict[str, dict[str, np.ndarray]]]:
#     with torch.no_grad():
#         features = {}
#         for img_size, images in multiscale_images.items():
#             # 打印输入图像的形状
#             print(f"Input image shape: {images.shape}")
#
#             # 获取原始特征
#             features_gem, features_clip = gem_model.model.visual(images.to(device))
#
#             # 创建FFC增强模块，使用实际的输入通道数
#             input_channels = images.shape[1]  # 获取实际的输入通道数
#             ffc_enhancer = FFC_BN_ACT(
#                 in_channels=input_channels,  # 使用实际的输入通道数
#                 out_channels=input_channels,  # 保持输出通道数相同
#                 kernel_size=3,
#                 ratio_gin=0.5,
#                 ratio_gout=0.5,
#                 padding=1,
#                 norm_layer=nn.BatchNorm2d,
#                 activation_layer=nn.ReLU,
#                 enable_lfu=True
#             ).to(device)
#
#             # 创建CBAM增强模块
#             cbam = EnhancedSimpleCBAM(
#                 in_channels=input_channels,
#                 feature_dim=features_gem.shape[-1],
#                 seq_length=features_gem.shape[1]
#             ).to(device)
#
#             # 获取FFC增强特征
#             enhanced_images = ffc_enhancer(images.to(device))
#
#             # 使用增强后的图像获取CBAM特征
#             enhanced_gem, enhanced_clip = cbam(enhanced_images)
#
#             # 特征融合
#             alpha = 0.6  # 原始特征权重
#             beta = 0.2  # FFC增强特征权重
#             gamma = 0.2  # CBAM增强特征权重
#
#             # 获取FFC处理后的特征
#             ffc_gem, ffc_clip = gem_model.model.visual(enhanced_images)
#
#             # 三路特征融合
#             features_gem = (alpha * features_gem +
#                             beta * ffc_gem +
#                             gamma * enhanced_gem)
#
#             features_clip = (alpha * features_clip +
#                              beta * ffc_clip +
#                              gamma * enhanced_clip)
#
#             # 标准化
#             features_gem = F.normalize(features_gem, dim=-1).cpu().numpy()
#             features_clip = F.normalize(features_clip, dim=-1).cpu().numpy()
#
#             features[img_size] = {
#                 "gem": {"cls": features_gem[:, 0, :], "patch": features_gem[:, 1:, :]},
#                 "clip": {"cls": features_clip[:, 0, :], "patch": features_clip[:, 1:, :]}
#             }
#
#             # 打印特征维度信息
#             print(f"Size {img_size}:")
#             print(f"GEM feature shape: {features_gem.shape}")
#             print(f"CLIP feature shape: {features_clip.shape}")
#
#     return features
# def extract_image_embeddings(       # 最高的
#         multiscale_images: dict[int, torch.Tensor],
#         gem_model,
#         device
# ) -> dict[int, dict[str, dict[str, np.ndarray]]]:
#     with torch.no_grad():
#         features = {}
#         for img_size, images in multiscale_images.items():
#
#             # 获取原始特征
#             features_gem, features_clip = gem_model.model.visual(images.to(device))
#
#             # 创建并应用增强模块
#             cbam = EnhancedSimpleCBAM(
#                 in_channels=3,
#                 feature_dim=features_gem.shape[-1],
#                 seq_length=features_gem.shape[1]
#             ).to(device)
#
#             # 获取增强特征
#             enhanced_gem, enhanced_clip = cbam(images.to(device))
#
#             # 特征融合
#             alpha = 0.93  # 可调节的权重参数
#             features_gem = alpha * features_gem + (1 - alpha) * enhanced_gem
#             features_clip = alpha * features_clip + (1 - alpha) * enhanced_clip
#
#             # 标准化
#             features_gem = F.normalize(features_gem, dim=-1).detach().cpu().numpy()
#             features_clip = F.normalize(features_clip, dim=-1).detach().cpu().numpy()
#
#             features[img_size] = {
#                 "gem": {"cls": features_gem[:, 0, :], "patch": features_gem[:, 1:, :]},
#                 "clip": {"cls": features_clip[:, 0, :], "patch": features_clip[:, 1:, :]}
#             }
#     return features
# 定义了一个名为 extract_image_embeddings 的函数，用于从多尺度图像中提取特征嵌入
# def extract_image_embeddings(
#         multiscale_images: dict[int, torch.Tensor], gem_model, device
# ) -> dict[int, dict[str, dict[str, np.ndarray]]]:
#     with torch.no_grad():
#         features = {}
#         for img_size, images in multiscale_images.items():
#             # 获取原始特征
#             features_gem, features_clip = gem_model.model.visual(images.to(device))
#
#             # 创建增强的SimpleCBAM
#             cbam = SimpleCBAM(
#                 in_channels=3,
#                 feature_dim=features_gem.shape[-1],
#                 seq_length=features_gem.shape[1]
#             ).to(device)
#
#             # 获取增强特征
#             enhanced_gem, enhanced_clip = cbam(images.to(device))
#
#
#             features_gem =  features_gem + enhanced_gem
#             features_clip = features_clip + enhanced_clip
#
#             # 标准化
#             features_gem = F.normalize(features_gem, dim=-1).detach().cpu().numpy()
#             features_clip = F.normalize(features_clip, dim=-1).detach().cpu().numpy()
#
#             features[img_size] = {
#                 "gem": {"cls": features_gem[:, 0, :], "patch": features_gem[:, 1:, :]},
#                 "clip": {"cls": features_clip[:, 0, :], "patch": features_clip[:, 1:, :]}
#             }
#     return features

# def extract_image_embeddings(
#         multiscale_images: dict[int, torch.Tensor], gem_model, device
# ) -> dict[int, dict[str, dict[str, np.ndarray]]]:
#     with torch.no_grad():
#         features = {}
#         for img_size, images in multiscale_images.items():
#             # 确保图像在正确的设备上
#             images = images.to(device)
#
#             # 初始化用于图像的SCSA模块
#             image_scsa = SCSA(
#                 in_channels=3,  # 图像输入通道数
#                 dim=1024,
#                 head_num=8,
#                 window_size=7,
#                 group_kernel_sizes=[3, 5, 7, 9],
#                 qkv_bias=True
#             ).to(device)
#
#             # 应用SCSA注意力机制到图像
#             enhanced_images = image_scsa(images)
#
#             # 使用增强后的图像提取特征
#             features_gem, features_clip = gem_model.model.visual(enhanced_images)
#
#             # 调整特征维度
#             B, N, D = features_gem.shape
#             H = int(math.sqrt(N))
#             W = H
#
#             # 初始化用于特征的SCSA模块
#             feature_scsa = SCSA(
#                 in_channels=D,  # 特征维度作为输入通道数
#                 dim=1024,
#                 head_num=8,
#                 window_size=7,
#                 group_kernel_sizes=[3, 5, 7, 9],
#                 qkv_bias=True
#             ).to(device)
#
#             # 重塑特征并应用SCSA
#             features_gem_4d = features_gem[:, 1:, :].reshape(B, H, W, D).permute(0, 3, 1, 2)
#             features_clip_4d = features_clip[:, 1:, :].reshape(B, H, W, D).permute(0, 3, 1, 2)
#
#             # 保存CLS token
#             features_gem_cls = features_gem[:, 0:1, :]
#             features_clip_cls = features_clip[:, 0:1, :]
#
#             # 应用SCSA到特征
#             enhanced_features_gem = feature_scsa(features_gem_4d)
#             enhanced_features_clip = feature_scsa(features_clip_4d)
#
#             # 重塑回原始形状
#             enhanced_features_gem = enhanced_features_gem.permute(0, 2, 3, 1).reshape(B, N - 1, D)
#             enhanced_features_clip = enhanced_features_clip.permute(0, 2, 3, 1).reshape(B, N - 1, D)
#
#             # 添加残差连接
#             features_gem_enhanced = torch.cat([
#                 features_gem_cls,
#                 features_gem[:, 1:, :] + enhanced_features_gem
#             ], dim=1)
#
#             features_clip_enhanced = torch.cat([
#                 features_clip_cls,
#                 features_clip[:, 1:, :] + enhanced_features_clip
#             ], dim=1)
#
#             # 标准化和后处理
#             features_gem = F.normalize(features_gem_enhanced, dim=-1).detach().cpu().numpy()
#             features_clip = F.normalize(features_clip_enhanced, dim=-1).detach().cpu().numpy()
#
#             features[img_size] = {
#                 "gem": {
#                     "cls": features_gem[:, 0, :],
#                     "patch": features_gem[:, 1:, :],
#                 },
#                 "clip": {
#                     "cls": features_clip[:, 0, :],
#                     "patch": features_clip[:, 1:, :],
#                 },
#             }
#     return features
# def extract_image_embeddings(
#     multiscale_images: dict[int, torch.Tensor], gem_model, device
# ) -> dict[int, dict[str, dict[str, np.ndarray]]]:   # multiscale_images: 一个多尺度图像的字典，键为图像尺寸，值为对应的图像张量。gem_model: 一个模型对象，包含视觉模型
#     # 初始化SCSA模块
#     scsa = SCSA(
#         dim=1024,  # 需要根据你的特征维度调整
#         head_num=8,
#         window_size=7,
#         group_kernel_sizes=[3, 5, 7, 9],
#         qkv_bias=True
#     ).to(device)
#     # 确保模型在评估模式
#     scsa.eval()
#     with torch.no_grad():
#         features = {}
#         for img_size, images in multiscale_images.items():
#             # 提取GEM和CLIP特征: 使用 gem_model 提取当前图像的 GEM 和 CLIP 特征
#             # 应用SCSA注意力机制
#             # 假设输入图像的shape为 [B, C, H, W]
#             # 确保图像在正确的设备上
#             images = images.to(device)
#
#             enhanced_images = scsa(images)
#
#             # 使用增强后的图像提取特征
#             features_gem, features_clip = gem_model.model.visual(enhanced_images)
#
#             # 调整特征维度以适应SCSA
#             # 假设features_gem和features_clip的形状是 [B, N, D]
#             B, N, D = features_gem.shape
#
#             # 将特征重塑为4D张量 [B, D, H, W]
#             H = int(math.sqrt(N))
#             W = H
#
#             features_gem_4d = features_gem[:, 1:, :].reshape(B, H, W, D).permute(0, 3, 1, 2)
#             features_clip_4d = features_clip[:, 1:, :].reshape(B, H, W, D).permute(0, 3, 1, 2)
#
#             # 保存CLS token
#             features_gem_cls = features_gem[:, 0:1, :]
#             features_clip_cls = features_clip[:, 0:1, :]
#
#             # 对重塑后的特征应用SCSA
#             enhanced_features_gem = scsa(features_gem_4d)
#             enhanced_features_clip = scsa(features_clip_4d)
#
#             # 将增强后的特征重塑回原始形状
#             enhanced_features_gem = enhanced_features_gem.permute(0, 2, 3, 1).reshape(B, N - 1, D)
#             enhanced_features_clip = enhanced_features_clip.permute(0, 2, 3, 1).reshape(B, N - 1, D)
#             # 添加残差连接
#             features_gem_enhanced = torch.cat([
#                 features_gem_cls,
#                 features_gem[:, 1:, :] + enhanced_features_gem
#             ], dim=1)
#
#             features_clip_enhanced = torch.cat([
#                 features_clip_cls,
#                 features_clip[:, 1:, :] + enhanced_features_clip
#             ], dim=1)
#
#             # 标准化和后处理
#             # ------------------------------------------------
#             # # 处理图像并保存原始输入的特征图
#             # processed_images = process_images_with_conv_layers(images)
#             # # 将原始输入转换为适合残差连接的形式
#             # # 这里假设原始输入和模型输出的通道数相同，否则需要进行调整
#             # x = F.interpolate(images, size=(processed_images.shape[2], processed_images.shape[3]), mode='bilinear',
#             #                   align_corners=False).to(device)
#             #
#             # # 提取GEM和CLIP特征
#             # features_gem, features_clip = gem_model.model.visual(processed_images.to(device))
#             #
#             # # 确保特征形状匹配，如果必要的话进行调整
#             # # 注意：这里假设 features_gem 和 features_clip 的形状一致
#             # if features_gem.shape != x.shape:
#             #     x = F.interpolate(x, size=(features_gem.shape[-2], features_gem.shape[-1]), mode='bilinear',
#             #                       align_corners=False)
#             #     # 选择第一个通道
#             #     x_channel_0 = x[:, 0:1, :, :]
#             #
#             #     # 挤压掉通道维度
#             #     x = x_channel_0.squeeze(1)
#             # # 应用残差连接
#             # features_gem += x
#             # features_clip += x
#            # ---------------------------------------------------------------
#
#             # features_gem, features_clip = gem_model.model.visual(processed_images.to(device))
#             # features_gem, features_clip = gem_model.model.visual(images.to(device))
#             # 归一化特征: 对提取的特征进行归一化处理.将归一化后的特征转换为 NumPy 数组。
#             features_gem = F.normalize(features_gem, dim=-1).detach().cpu().numpy()
#             features_clip = F.normalize(features_clip, dim=-1).detach().cpu().numpy()
#             # 组织特征字典: 将特征组织成一个嵌套字典结构，包含每个图像尺寸下的 GEM 和 CLIP 特征的 cls 和 patch 部分
#             features[img_size] = {
#                 "gem": {
#                     "cls": features_gem[:, 0, :],
#                     "patch": features_gem[:, 1:, :],
#                 },
#                 "clip": {
#                     "cls": features_clip[:, 0, :],
#                     "patch": features_clip[:, 1:, :],
#                 },
#             }
#     return features

# 定义了一个名为 retrieve_image_embeddings 的函数，用于从给定的图像嵌入字典中提取特定的嵌入向量。
def retrieve_image_embeddings(
    image_embeddings: dict[int, dict[str, dict[str, np.ndarray]]],
    img_size: int,
    feature_type: str,
    token_type: str,
) -> np.ndarray:
    """
    Args:
        image_embeddings: All image embeddings.
        img_size: integer indicating image size
        feature_type: 'clip' or 'gem'
        token_type: 'cls' or 'patch'

    Returns:
        Numpy array of dimension
        (batch_size, emb_dim) when token_type='cls'
        (batch_size, num_patches, emb_dim) when token_type='patch'
        当 token_type 为 'cls' 时，返回形状为 (batch_size, emb_dim) 的 NumPy 数组。
        当 token_type 为 'patch' 时，返回形状为 (batch_size, num_patches, emb_dim) 的 NumPy 数组。
    """
    return image_embeddings[img_size][feature_type][token_type]
