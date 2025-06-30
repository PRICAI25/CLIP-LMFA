import torch
import torch.nn as nn
import numpy as np

# def init_weight(module):
#     if isinstance(module, nn.Linear):
#         nn.init.xavier_uniform_(module.weight)
#         if module.bias is not None:
#             nn.init.constant_(module.bias, 0)
#     elif isinstance(module, nn.LayerNorm):
#         nn.init.constant_(module.weight, 1)
#         nn.init.constant_(module.bias, 0)
#
# class MLP(nn.Module):
#     def __init__(self, in_dim, hid_dim, act_fn, drop):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(in_dim, hid_dim)
#         self.act = act_fn()
#         self.fc2 = nn.Linear(hid_dim, in_dim)
#         self.dropout = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
# class ImageHead_image(nn.Module):
#     '''
#     Image head module adapted for image-to-image feature processing.
#
#     Args:
#         feature_dim (int): Input feature dimension.
#         mlp_layers (int): Number of MLP layers.
#         mlp_ratio (int): Multiplier to compute the hidden layer dimension.
#         out_dim (int): Output dimension.
#         act_fn (nn.Module): Activation function.
#         drop (float): Dropout probability.
#         res (bool): Whether to use residual connection.
#         no_head (bool): Whether to bypass the head module.
#         project_out (bool): Whether to project the output.
#     '''
#
#     def __init__(self, feature_dim, mlp_layers=2, mlp_ratio=4, out_dim=256, act_fn=nn.GELU, drop=0., res=False,
#                  no_head=False, project_out=True):
#         super(ImageHead_image, self).__init__()
#         self.mlp_layers = mlp_layers
#         self.no_head = no_head
#
#         # MLP layers for feature transformation
#         self.mlps = nn.ModuleList([
#             MLP(feature_dim, feature_dim * mlp_ratio, act_fn, drop)
#             for _ in range(mlp_layers)
#         ])
#
#         # Attention parameters
#         self.scale = feature_dim ** -0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(drop)
#
#         # Final projection
#         self.project_out = project_out
#         if project_out:
#             self.to_out = nn.Sequential(
#                 nn.Linear(feature_dim, out_dim),
#                 nn.Dropout(drop)
#             )
#
#         # Residual connection
#         self.res = res
#         self.norm = nn.LayerNorm(feature_dim)
#
#         # Apply weight initialization
#         self.apply(init_weight)
#
#     def forward(self, query_feature, memory_feature):
#         if self.no_head:
#             return query_feature
#         query_feature = torch.from_numpy(query_feature)
#         memory_feature = torch.from_numpy(memory_feature)
#         # Normalize features
#         query_feature = self.norm(query_feature)
#         memory_feature = self.norm(memory_feature)
#
#         # Compute attention
#         attended_feature = self.image_to_image_attention(query_feature, memory_feature)
#
#         # Apply MLPs
#         residual = query_feature
#         for mlp_layer in self.mlps:
#             attended_feature = mlp_layer(attended_feature)
#
#         # Residual connection
#         if self.res:
#             attended_feature = attended_feature + residual
#
#         # Final projection
#         if self.project_out:
#             attended_feature = self.to_out(attended_feature)
#
#
#         attended_feature = attended_feature.detach().numpy()
#         return attended_feature
#
#     def image_to_image_attention(self, query_feature, memory_feature):
#         # Compute attention scores
#         dots = torch.matmul(query_feature, memory_feature.transpose(-2, -1)) * self.scale
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         # Apply attention to memory features
#         out = torch.matmul(attn, memory_feature)
#         return out
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weight(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, act_fn, drop):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act_fn()
        self.fc2 = nn.Linear(hid_dim, in_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class ImageHead_image(nn.Module):
    def __init__(self, feature_dim, mlp_layers=4, mlp_ratio=8, out_dim=256, act_fn=nn.GELU, drop=0.1, res=True,
                 no_head=False, project_out=True):
        super(ImageHead_image, self).__init__()
        self.mlp_layers = mlp_layers
        self.no_head = no_head
        self.mlps = nn.ModuleList([
            MLP(feature_dim, feature_dim * mlp_ratio, act_fn, drop)
            for _ in range(mlp_layers)
        ])
        self.scale = feature_dim ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop)
        self.project_out = project_out
        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(feature_dim, out_dim),
                nn.Dropout(drop)
            )
        self.res = res
        self.norm = nn.LayerNorm(feature_dim)
        self.apply(init_weight)

    def forward(self, query_feature, memory_feature):
        global attended_feature_out
        if self.no_head:
            return query_feature
        # Normalize features
        residual = query_feature
        query_feature = torch.tensor(query_feature, dtype=torch.float32)
        memory_feature = torch.tensor(memory_feature, dtype=torch.float32)
        query_feature = self.norm(query_feature)
        memory_feature = self.norm(memory_feature)
        attended_feature = self.image_to_image_attention(query_feature, memory_feature)

        for mlp_layer in self.mlps:
            attended_feature = mlp_layer(attended_feature)

        if self.project_out:
                attended_feature_out = self.to_out(attended_feature)
        if self.res:
            alpha = nn.Parameter(torch.tensor(1.0))  # 可学习权重
            beta = nn.Parameter(torch.tensor(1.0))  # 可学习权重
            gamma = nn.Parameter(torch.tensor(1.0))  # 可学习权重
            attended_feature =  gamma * attended_feature_out + beta * residual +  alpha * attended_feature
            attended_feature = F.relu(attended_feature)  # 添加激活函数

        return attended_feature.detach().numpy()

    def image_to_image_attention(self, query_feature, memory_feature):
        dots = torch.matmul(query_feature, memory_feature.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, memory_feature)
        return out
#
# class ImageModel:
#     def __init__(self, kth: int = 1, feature_dim: int = 640):
#         self.kth = kth
#         self.image_head = ImageHead_image(feature_dim=feature_dim, out_dim=feature_dim)
#
#     def fit(self, train_image_embeddings: np.ndarray):
#         self.train_image_embeddings = train_image_embeddings
#
#     def predict_proba(self, image_embeddings: np.ndarray) -> np.ndarray:
#         num_patches, emb_dim = image_embeddings.shape
#         with torch.no_grad():
#             train_image_embeddings = self.image_head(image_embeddings, self.train_image_embeddings)
#         train_image_embeddings = 0.8 * self.train_image_embeddings + 0.2 * train_image_embeddings
#         cosine_sim = image_embeddings @ train_image_embeddings.T
#         anomaly_scores = 0.5 * (1 - cosine_sim)
#         return np.partition(anomaly_scores, self.kth - 1, axis=1)[:, self.kth - 1]