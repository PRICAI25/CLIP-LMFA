from bisect import bisect
from typing import Union, List

import numpy as np
import pandas as pd

from evaluation import metrics
from evaluation.metrics import compute_pro

# 定义了一个名为 trapezoid 的函数，用于计算给定 x 和 y 值的曲线下的面积。
def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    # 检查 x 和 y 中是否有非有限值，如果有，则发出警告并过滤掉这些值。
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            "WARNING: Not all x and y values passed to trapezoid(...)"
            " are finite. Will continue with only the finite values."
        )
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            # 获取插入索引，使得在插入 x_max 后 x 仍然保持排序。
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            # x_max 必须在最小值和最大值之间，因此插入点不能是 0 或 len(x)。
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            # 计算校正项，即 x[ins-1] 和 x_max 之间的积分。由于我们不知道 x_max 处的确切 y 值，
            # 我们在 y[ins] 和 y[ins-1] 之间进行插值。
            y_interp = y[ins - 1] + (
                (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            )
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        # 在 x_max 处截断。 创建一个布尔掩码 mask，使得 x 和 y 只保留小于等于 x_max 的部分
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    # 使用梯形法则返回曲线下面积。
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction

# 定义了一个名为 compute_and_store_final_results 的函数，其主要功能是计算一组结果的平均值，
# 并将这些结果和平均值存储在一个 Pandas DataFrame 中
def compute_and_store_final_results(results: List[dict]) -> pd.DataFrame:
    # 创建一个字典 mean_metrics，并设置初始键 "object_name" 的值为 "mean"
    mean_metrics = {"object_name": "mean"}
    # 从 results 列表的最后一个字典中提取所有指标名称（除第一个键 "object_name" 外）
    metric_names = list(results[-1].keys())[1:]
    # 将平均值存储到 mean_metrics：将计算得到的平均值存储到 mean_metrics 字典中
    for i, result_key in enumerate(metric_names):
        mean_metrics[result_key] = np.mean([x[result_key] for x in results])

    header = list(results[-1].keys())
    # 将 results 列表和 mean_metrics 字典合并，并转换为一个 Pandas DataFrame。
    dt = pd.DataFrame(results + [mean_metrics], columns=header)
    return dt

# 定义了一个名为 evaluation 的函数，用于评估模型在图像异常检测任务中的性能
def evaluation(
    ground_truth_labels: Union[list, np.array],
    predicted_labels: Union[list, np.array],
    ground_truth_segmentations: Union[List[np.array], np.array],
    predicted_segmentations: Union[List[np.array], np.array],
    integration_limit: float = 0.3,
) -> dict:
    full_image_auroc = np.nan
    full_image_aupr = np.nan
    full_image_f1_max = np.nan
    full_pixel_auroc = np.nan
    full_pixel_f1_max = np.nan
    full_pixel_au_pro = np.nan
    anomaly_pixel_auroc = np.nan
    anomaly_pixel_f1_max = np.nan
    # 如果 ground_truth_labels 和 predicted_labels 都不为空，则计算图像级别的 AUROC、AUPR 和 F1 分数
    if len(ground_truth_labels) != 0 and len(predicted_labels) != 0:
        # Compute image-level Auroc for all images
        image_scores = metrics.compute_imagewise_retrieval_metrics(
            predicted_labels, ground_truth_labels
        )
        full_image_auroc = image_scores["auroc"]
        full_image_aupr = image_scores["aupr"]
        full_image_f1_max = image_scores["f1_max"]
    # 如果 ground_truth_segmentations 和 predicted_segmentations 都不为空，则计算像素级别的 AUROC、F1 分数和 PRO 曲线下的面积
    if len(ground_truth_segmentations) != 0 and len(predicted_segmentations) != 0:
        # Compute PW Auroc for all images
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            predicted_segmentations, ground_truth_segmentations
        )
        full_pixel_auroc = pixel_scores["auroc"]
        full_pixel_f1_max = pixel_scores["f1_max"]
        # 存在包含异常的图像？
        pro_curve = compute_pro(
            anomaly_maps=predicted_segmentations,
            ground_truth_maps=ground_truth_segmentations,
        )

        # Compute the area under the PRO curve.
        full_pixel_au_pro = trapezoid(
            pro_curve[0], pro_curve[1], x_max=integration_limit
        )
        full_pixel_au_pro /= integration_limit

        # Compute PRO score & PW Auroc only for images with anomalies
        # 计算异常像素的 AUROC 和 F1 分数

        sel_idxs = []
        for i in range(len(ground_truth_segmentations)):
            if np.sum(ground_truth_segmentations[i]) > 0:
                sel_idxs.append(i)
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            [predicted_segmentations[i] for i in sel_idxs],
            [ground_truth_segmentations[i] for i in sel_idxs],
        )
        anomaly_pixel_auroc = pixel_scores["auroc"]
        anomaly_pixel_f1_max = pixel_scores["f1_max"]

    return {
        "full_image_auroc": full_image_auroc,
        "full_image_aupr": full_image_aupr,
        "full_image_f1_max": full_image_f1_max,
        "full_pixel_auroc": full_pixel_auroc,
        "full_pixel_f1_max": full_pixel_f1_max,
        "full_pixel_au_pro": full_pixel_au_pro,
        "anomaly_pixel_auroc": anomaly_pixel_auroc,
        "anomaly_pixel_f1_max": anomaly_pixel_f1_max,
    }
