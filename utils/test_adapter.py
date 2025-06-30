import copy
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.SCSA_two import EnhancedSimpleCBAM
#
# IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
# IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]
# # 数据预处理
# # def train_val_data_process(multiscale_images):
# #     all_images = []
# #
# #     for img_size, images in multiscale_images.items():
# #         all_images.extend(images)  # 将每个图像列表中的图像添加到 all_images 中
# #
# #     normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
# #
# #     train_transform = transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         transforms.ToTensor(),
# #         normalize
# #     ])
# #
# #     train_data = ImageFolder(all_images, transform=train_transform)
# #     # train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
# #
# #     train_dataloader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)
# #     val_dataloader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)
# #
# #     return train_dataloader, val_dataloader
#
#
# def train_model_process(model, train_dataloader,  num_epochs, ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#     model = model.to(device)
#     # 在你的代码中找到 train_dataloader 被定义的地方
#     # 然后添加如下代码来打印其结构和值
#
#     try:
#         batch = next(iter(train_dataloader))
#         print("Batch structure and values:")
#         print(batch)
#     except StopIteration:
#         print("The dataloader is empty.")
#
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     train_loss_all = []
#     val_loss_all = []
#     train_acc_all = []
#     val_acc_all = []
#     since = time.time()
#
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch}/{num_epochs - 1}")
#         print("-" * 10)
#
#         train_loss = 0.0
#         train_corrects = 0
#         train_num = 0
#
#         # 训练阶段
#         model.train()
#         for i, data in enumerate(ref_dataloader):
#             multiscale_images = {sz: data["image"][sz] for sz in img_sizes}
#
#         for step, (b_x, b_y) in enumerate(train_dataloader):
#             b_x = b_x.to(device)
#             b_y = b_y.to(device)
#
#             optimizer.zero_grad()
#             output = model(b_x)
#             loss = criterion(output, b_y)
#
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item() * b_x.size(0)
#             train_corrects += torch.sum(torch.argmax(output, dim=1) == b_y.data)
#             train_num += b_x.size(0)
#
#
#         # 计算并保存每一次迭代的loss值和准确率
#         train_loss_all.append(train_loss / train_num)
#         train_acc_all.append(train_corrects.double().item() / train_num)
#
#         print(f"Train Loss: {train_loss_all[-1]:.4f} Train Acc: {train_acc_all[-1]:.4f}")
#
#         # 寻找最高准确度的权重
#         if train_acc_all[-1] > best_acc:  # 使用训练集准确度
#             best_acc = train_acc_all[-1]
#             best_model_wts = copy.deepcopy(model.state_dict())
#         time_use = time.time() - since
#         print(f"Training and validation time: {time_use // 60:.0f}m {time_use % 60:.0f}s")
#     torch.save(model.state_dict(best_model_wts), '/home/wsc/BMVC-FADE-main/scripts/best_model.pth')
#
#
# # 画图
# # def matplot_acc_loss(train_process):
# #     plt.figure(figsize=(12, 4))
# #     plt.subplot(1, 2, 1)
# #     plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss")
# #     plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
# #     plt.legend()
# #     plt.xlabel("epoch")
# #     plt.ylabel("loss")
# #
# #     plt.subplot(1, 2, 2)
# #     plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc")
# #     plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
# #     plt.legend()
# #     plt.xlabel("epoch")
# #     plt.ylabel("acc")
# #     plt.show()
#
# def train_model(in_channels, feature_dim, seq_length, train_dataloader):
#     model = EnhancedSimpleCBAM(in_channels=in_channels, feature_dim=feature_dim, seq_length=seq_length)  # 根据需要调整参数
#     train_model_process(model, train_dataloader, num_epochs=20)
#     # 加载最佳模型参数
#     # model.load_state_dict(best_model_wts)

# if __name__ == "__main__":
#     # 实例化模型
#     model = EnhancedSimpleCBAM(in_channels=3, feature_dim=256, seq_length=10, multiscale_images)  # 根据需要调整参数
#     train_dataloader, val_dataloader = train_val_data_process(multiscale_images)
#     train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=20)
#     matplot_acc_loss(train_process)




import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming EnhancedSimpleCBAM and ref_dataloader are already defined

# def train_model_process(model, ref_dataloader, num_epochs, device):
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # L2 regularization to prevent overfitting
#     best_acc = 0.0
#     best_model_path = 'best_model.pth'
#     supported_sizes = [448 , 896]
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
#
#         for i, data in enumerate(ref_dataloader):
#             # Combine all multiscale images into a single batch
#             multiscale_images = {sz: data["image"][sz] for sz in supported_sizes}
#             for img_size, images in multiscale_images.items():
#                 images = images.to(device)
#                 labels = data["is_anomaly"].to(device)  # Assuming 'is_anomaly' is the label
#                 if labels.dim() > 1:
#                     labels = labels.squeeze()  # Remove any extra dimensions
#
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 if isinstance(outputs, tuple):
#                     outputs = outputs[0]  # Assuming the first element is the logits
#                 outputs = outputs[:, 0, :]  # Select the first set of logits
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 running_loss += loss.item() * images.size(0)  # Use images.size(0) for batch size
#                 _, preds = torch.max(outputs, 1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#
#         epoch_loss = running_loss / len(ref_dataloader.dataset)
#         epoch_acc = accuracy_score(all_labels, all_preds)
#
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
#
#         # Save the best model
#         if epoch_acc > best_acc:
#             best_acc = epoch_acc
#             torch.save(model.state_dict(), best_model_path)
#             print(f'Saved best model with accuracy: {best_acc:.4f}')
#
#     print('Training complete. Best accuracy: {:.4f}'.format(best_acc))



import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_240, features_448, features_896):
        # 将896尺度的特征下采样到448尺度
        features_896_down = F.interpolate(features_896, size=features_448.shape[-2:], mode='bilinear', align_corners=True)
        # 将240尺度的特征上采样到448尺度
        features_240_up = F.interpolate(features_240, size=features_448.shape[-2:], mode='bilinear', align_corners=True)
        # 将特征拼接
        fused_features = torch.cat([features_240_up, features_448, features_896_down], dim=1)
        # 通过卷积融合
        fused_features = self.fusion(fused_features)
        return fused_features

# 加权
# class MultiScaleFeatureFusion(nn.Module):
#     def __init__(self, in_channels):
#         super(MultiScaleFeatureFusion, self).__init__()
#         self.fusion = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.residual_connection = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
#         self.alpha = nn.Parameter(torch.tensor(0.1))  # 可学习权重
#         self.beta = nn.Parameter(torch.tensor(0.1))   # 可学习权重
#         self.gamma = nn.Parameter(torch.tensor(0.1))  # 可学习权重
#
#     def forward(self, features_240, features_448, features_896):
#         features_896_down = F.interpolate(features_896, size=features_448.shape[-2:], mode='bilinear', align_corners=True)
#         features_240_up = F.interpolate(features_240, size=features_448.shape[-2:], mode='bilinear', align_corners=True)
#         fused_features = torch.cat([features_448, features_896_down], dim=1)
#
#         residual = self.residual_connection(fused_features)
#         fused_features = self.fusion(fused_features)
#
#         # 动态加权残差连接
#         fused_features =  fused_features + residual + self.gamma * features_240_up
#
#         return fused_features

# 基本
# class MultiScaleFeatureFusion(nn.Module):
#     def __init__(self, in_channels):
#         super(MultiScaleFeatureFusion, self).__init__()
#         self.fusion = nn.Sequential(
#             nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.residual_connection = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
#
#     def forward(self, features_240, features_448, features_896):
#         # 类似地进行上/下采样及特征拼接
#         features_896_down = F.interpolate(features_896, size=features_448.shape[-2:], mode='bilinear',
#                                           align_corners=True)
#         features_240_up = F.interpolate(features_240, size=features_448.shape[-2:], mode='bilinear', align_corners=True)
#         fused_features = torch.cat([features_240_up, features_448, features_896_down], dim=1)
#
#         # 加入残差连接
#         residual = self.residual_connection(fused_features)
#         fused_features = self.fusion(fused_features) + residual
#
#         return fused_features


class MultiScaleFeatureFusion_1(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureFusion_1, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_240, features_448):
        # 将896尺度的特征下采样到448尺度

        # 将240尺度的特征上采样到448尺度
        features_240_up = F.interpolate(features_240, size=features_448.shape[-2:], mode='bilinear', align_corners=True)
        # 将特征拼接
        fused_features = torch.cat([features_240_up, features_448], dim=1)
        # 通过卷积融合
        fused_features = self.fusion(fused_features)
        return fused_features
# 三合一的
def train_model_process(model, ref_dataloader, num_epochs, device, shots):
    model.to(device)
    patience = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    best_acc = 0.0
    best_model_path = 'btad-best_model.pth'
    supported_sizes = [240, 448, 896]
    # supported_sizes = [240, 448]
    no_improvement = 0

    # 定义特征融合模块
    feature_fusion = MultiScaleFeatureFusion(in_channels=3).to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for i, data in enumerate(ref_dataloader):
            # Combine all multiscale images into a single batch
            multiscale_images = {sz: data["image"][sz] for sz in supported_sizes}
            labels = data["is_anomaly"].to(device)
            if labels.dim() > 1:
                labels = labels.squeeze()  # Remove any extra dimensions

            optimizer.zero_grad()

            # 提取每个尺度的特征
            features_240 = multiscale_images[240].to(device)
            features_448 = multiscale_images[448].to(device)
            features_896 = multiscale_images[896].to(device)



            # 将特征拼接
            fused_features = feature_fusion(features_240, features_448, features_896)


            # 将融合后的特征输入到模型中
            outputs = model(fused_features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Assuming the first element is the logits
            outputs = outputs[:, 0, :]  # Select the first set of logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1 == shots):
                break

        epoch_loss = running_loss / len(ref_dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Save the best model
        if epoch_acc > best_acc or best_acc == 0.0:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with accuracy: {best_acc:.4f}')
            no_improvement = 0
        else:
            no_improvement += 1
        # Early stopping
        if no_improvement >= patience:

            print(f'Early stopping triggered after {patience} epochs without improvement.')
            break

    print('Training complete. Best accuracy: {:.4f}'.format(best_acc))

# 相加两层的
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import accuracy_score
#
# def train_model_process(model, ref_dataloader, num_epochs, device,  shots):
#     model.to(device)
#     patience = 2
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
#     best_acc = 0.0
#     best_model_path = 'btad-best_model.pth'       #  'visa-best_model.pth'
#     supported_sizes = [448, 896]
#     no_improvement = 0
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
#
#         for i, data in enumerate(ref_dataloader):
#             # Combine all multiscale images into a single batch
#
#             print(data)
#             multiscale_images = {sz: data["image"][sz] for sz in supported_sizes}
#
#             labels = data["is_anomaly"].to(device)
#             if labels.dim() > 1:
#                 labels = labels.squeeze()  # Remove any extra dimensions
#
#             optimizer.zero_grad()
#
#             # 初始化 image 变量
#             image = None
#             for img_size, images in multiscale_images.items():
#                 images = images.to(device)
#                 if image is None:
#                     image = images.clone()  # 初始化 image 为当前图像张量的副本
#                 else:
#                     # 确保图像形状一致
#                     if img_size == 896:
#                         # 将896尺度的图像下采样到448尺度
#                         images = F.interpolate(images, size=(448, 448), mode='bilinear', align_corners=True)
#                     image = image + images
#
#             # 将加和后的图像输入到模型中
#             outputs = model(image)
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]  # Assuming the first element is the logits
#             outputs = outputs[:, 0, :]  # Select the first set of logits
#
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item() * labels.size(0)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#         epoch_loss = running_loss / len(ref_dataloader.dataset)
#         epoch_acc = accuracy_score(all_labels, all_preds)
#
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
#
#         # Save the best model
#         if epoch_acc > best_acc or best_acc == 0.0:
#             best_acc = epoch_acc
#             torch.save(model.state_dict(), best_model_path)
#             print(f'Saved best model with accuracy: {best_acc:.4f}')
#         else:
#             no_improvement += 1
#         # Early stopping
#         if no_improvement >= patience:
#             print(f'Early stopping triggered after {patience} epochs without improvement.')
#             break
#
#         if (i + 1 == shots):
#             break
#     print('Training complete. Best accuracy: {:.4f}'.format(best_acc))
# 最高
# def train_model_process(model, ref_dataloader, num_epochs, device, shots):
#     model.to(device)
#     patience = 2
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # L2 regularization to prevent overfitting
#     best_acc = 0.0
#     best_model_path = 'best_model.pth'
#     supported_sizes = [448, 896]
#     no_improvement = 0  # 计数器，用于跟踪没有改进的周期数
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
#
#         for i, data in enumerate(ref_dataloader):
#             # Combine all multiscale images into a single batch
#             multiscale_images = {sz: data["image"][sz] for sz in supported_sizes}
#             for img_size, images in multiscale_images.items():
#                 images = images.to(device)
#                 labels = data["is_anomaly"].to(device)  # Assuming 'is_anomaly' is the label
#                 if labels.dim() > 1:
#                     labels = labels.squeeze()  # Remove any extra dimensions
#
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 if isinstance(outputs, tuple):
#                     outputs = outputs[0]  # Assuming the first element is the logits
#                 outputs = outputs[:, 0, :]  # Select the first set of logits
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 running_loss += loss.item() * images.size(0)  # Use images.size(0) for batch size
#                 _, preds = torch.max(outputs, 1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#
#             if(i + 1 == shots):
#                 break
#
#         epoch_loss = running_loss / len(ref_dataloader.dataset)
#         epoch_acc = accuracy_score(all_labels, all_preds)
#
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
#
#         # Save the best model
#         if epoch_acc > best_acc or best_acc == 0.0:
#             best_acc = epoch_acc
#             torch.save(model.state_dict(), best_model_path)
#             print(f'Saved best model with accuracy: {best_acc:.4f}')
#         else:
#             no_improvement += 1
#         # Early stopping
#         if no_improvement >= patience:
#             no_improvement = 0
#             print(f'Early stopping triggered after {patience} epochs without improvement.')
#             break
#
#     print('Training complete. Best accuracy: {:.4f}'.format(best_acc))




# def train_model_process(model, ref_dataloader, num_epochs, device, shots):
#     model.to(device)
#     patience = 2
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  # L2 regularization to prevent overfitting
#     best_acc = 0.0
#     best_model_path = 'best_model.pth'
#     supported_sizes = [240, 448, 896]
#     no_improvement = 0  # 计数器，用于跟踪没有改进的周期数
#
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         all_preds = []
#         all_labels = []
#
#         for i, data in enumerate(ref_dataloader):
#             # Combine all multiscale images into a single batch
#             multiscale_images = {sz: data["image"][sz] for sz in supported_sizes}
#             labels = data["is_anomaly"].to(device)
#             if labels.dim() > 1:
#                 labels = labels.squeeze()  # Remove any extra dimensions
#
#             optimizer.zero_grad()
#             target_size = (448, 448)  # 选择一个目标大小
#             adjusted_images = []
#             for sz in supported_sizes:
#                 img = multiscale_images[sz]
#                 if sz != target_size[0]:
#                     img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=True)
#                 adjusted_images.append(img)
#
#             batch_images = torch.cat(adjusted_images, dim=0)
#             batch_images = batch_images.to(device)
#
#             outputs = model(batch_images)
#             if isinstance(outputs, tuple):
#                 outputs = outputs[0]  # Assuming the first element is the logits
#             outputs = outputs[:, 0, :]  # Select the first set of logits
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item() * batch_images.size(0)  # Use batch size
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#             if (i + 1 == shots):
#                 break
#
#         epoch_loss = running_loss / len(ref_dataloader.dataset)
#         epoch_acc = accuracy_score(all_labels, all_preds)
#
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
#
#         # Save the best model
#         if epoch_acc > best_acc or best_acc == 0.0:
#             best_acc = epoch_acc
#             torch.save(model.state_dict(), best_model_path)
#             print(f'Saved best model with accuracy: {best_acc:.4f}')
#             no_improvement = 0  # Reset counter when accuracy improves
#         else:
#             no_improvement += 1
#
#         # Early stopping
#         if no_improvement >= patience:
#             print(f'Early stopping triggered after {patience} epochs without improvement.')
#             break
#
#     print('Training complete. Best accuracy: {:.4f}'.format(best_acc))
def load_model_weights(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    print(f'Model weights loaded from {weight_path}')

def train_model(ref_dataloader, device, shots):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedSimpleCBAM(in_channels=3, feature_dim=640, seq_length=226)  # 226 Adjust feature_dim and seq_length as needed
    train_model_process(model, ref_dataloader, 8, device, shots=shots)



# Example usage
