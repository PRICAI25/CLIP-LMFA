from typing import Union, List

import numpy as np
from scipy.ndimage.measurements import label as measurements_label
from sklearn import metrics

# 定义了一个名为 compute_imagewise_retrieval_metrics 的函数，用于计算图像级别的异常检测指标，
# 包括 AUROC（Area Under the ROC Curve）、FPR（False Positive Rate）、TPR（True Positive Rate）、AUPR（Area Under the Precision-Recall Curve）和最大 F1 分数
def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights: Union[list, np.array], # anomaly_prediction_weights: 图像的异常预测权重，值越高表示越可能是异常
    anomaly_ground_truth_labels: Union[list, np.array], # anomaly_ground_truth_labels: 图像的真实标签，1 表示异常，0 表示正常
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    # 计算 ROC 曲线：使用 metrics.roc_curve 计算 FPR 和 TPR
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    # 计算 AUROC：使用 metrics.roc_auc_score 计算 AUROC
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    # 计算 Precision-Recall 曲线：使用 metrics.precision_recall_curve 计算 Precision 和 Recall
    precision, recall, thresholds = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    # 计算 AUPR：使用 metrics.auc 计算 AUPR
    aupr = metrics.auc(recall, precision)
    # 计算 F1 分数：计算每个阈值下的 F1 分数，并找到最大值
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    # 一个字典，包含计算得到的 AUROC、FPR、TPR、AUPR 和最大 F1 分数
    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "aupr": aupr,
        "f1_max": max(F1_scores),
    }


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations: Union[List[np.array], np.array],
    ground_truth_masks: Union[List[np.array], np.array],
):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    # 检查输入的 anomaly_segmentations 和 ground_truth_masks 是否为列表，如果是，则将其转换为 NumPy 数组
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)
    # 将 anomaly_segmentations 和 ground_truth_masks 展平为一维数组，以便后续计算
    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()
    # 使用 metrics.roc_curve 计算 FPR、TPR 和阈值。
    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    # 使用 metrics.roc_auc_score 计算 AUROC
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    # 使用 metrics.precision_recall_curve 计算 Precision、Recall 和阈值。

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    # 计算 F1 分数，处理除零错误
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    # 确定最优阈值：找到 F1 分数最大时的阈值
    optimal_threshold = thresholds[np.argmax(F1_scores)]
    # 根据最优阈值生成预测结果。
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    # 计算 FPR 和 FNR
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "f1_max": max(F1_scores),
    }


def compute_pro(anomaly_maps: List[np.array], ground_truth_maps: List[np.array]):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """
    # 将输入的 anomaly_maps 和 ground_truth_maps 从列表转换为NumPy数组
    if isinstance(anomaly_maps, list):
        anomaly_maps = np.stack(anomaly_maps)
    if isinstance(ground_truth_maps, list):
        ground_truth_maps = np.stack(ground_truth_maps)
    # 如果 ground_truth_maps 是4维数组，将其转换为3维数组
    if len(ground_truth_maps.shape) == 4:
        ground_truth_maps = ground_truth_maps[:, 0, :, :]

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)
    # 初始化 num_ok_pixels 和 num_gt_regions 为0。

    num_ok_pixels = 0
    num_gt_regions = 0
    # 创建 fp_changes 和 pro_changes 数组，用于存储每个像素的FPR和PRO变化量
    shape = (len(anomaly_maps), anomaly_maps[0].shape[0], anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert (
        shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max
    ), "Potential overflow when using np.cumsum(), consider using np.uint64."

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map. 对每个地面真实图，计算连通区域
        labeled, n_components = measurements_label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all good pixels. 计算正常像素数和连通区域数
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        # 计算每个像素的FPR和PRO变化量，并存储在 fp_changes 和 pro_changes 中
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1.0 / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    # 将 anomaly_maps、fp_changes 和 pro_changes 展平为1维数组
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores. 按异常分数降序排序，并更新 fp_changes 和 pro_changes
    print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    # np.take(a, ind, out=a)：这个函数从数组 a 中选取索引为 ind 的元素，并将结果直接写回到数组 a 中，而不是创建一个新的数组。
    # b=a：这一步将引用 a 赋值给 b，而不是复制数组。
    # 直接使用 b=a[ind] 会创建一个新的数组 b，这可能会消耗更多的内存
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values. 使用 np.cumsum 计算累积的FPR和PRO值。归一化FPR和PRO值
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]

    # 合并在同一阈值下同时出现的 (FPR, PRO) 点。
    # 对于这些点，只保留最后一个 (FPR, PRO) 点。
    # 这是因为该点考虑了在相应阈值下的所有 FPR 和 PRO 的变化。
    # -> keep_mask 在后续得分与当前位置得分不同的情况下为 True。
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    # 为了减少在调用 np.cumsum 时累积的数值误差，
    # 确保曲线以 (1, 1) 结束，并且不包含大于 1 的值。
    np.clip(fprs, a_min=None, a_max=1.0, out=fprs)
    np.clip(pros, a_min=None, a_max=1.0, out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    # 使 FPR 和 PRO 从 0 开始并以 1 结束。
    zero = np.array([0.0])
    one = np.array([1.0])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))
