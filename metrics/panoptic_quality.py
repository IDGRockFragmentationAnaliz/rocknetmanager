import cv2
import numpy as np


def _closed_objects(edges: np.ndarray) -> np.ndarray:
    # 4-связность фона не пропускает его через диагональные 8-связные линии.
    count, components = cv2.connectedComponents(
        (edges == 0).astype(np.uint8), connectivity=4,
    )
    border_ids = np.unique(np.concatenate((
        components[0, :], components[-1, :],
        components[:, 0], components[:, -1],
    )))
    closed = np.ones(count, dtype=bool)
    closed[0] = False
    closed[border_ids] = False
    object_ids = np.zeros(count, dtype=np.int32)
    object_ids[closed] = np.arange(1, np.count_nonzero(closed) + 1)
    return object_ids[components]


def panoptic_quality(pred: np.ndarray, gt: np.ndarray) -> float:
    """PQ замкнутых объектов внутри 8-связных бинарных границ.

    Входы: непустые 2D-массивы np.uint8 одинаковой формы; 0 — фон,
    1 или 255 — граница. Границы и области, выходящие на край изображения,
    не входят в объекты. Исключённую маской область следует заранее
    заполнить границей на обеих картах. Входные массивы не изменяются.

    PQ = sum(IoU) / (TP + 0.5 * FP + 0.5 * FN), пары имеют IoU > 0.5.
    Если объектов нет на обеих картах, результат 1.0; на одной — 0.0.
    """
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("Обе карты границ должны быть двумерными")
    if pred.shape != gt.shape:
        raise ValueError(f"Размеры карт не совпадают: pred={pred.shape}, gt={gt.shape}")
    if pred.dtype != np.uint8 or gt.dtype != np.uint8:
        raise TypeError("Обе карты должны иметь dtype=np.uint8")
    if pred.size == 0:
        raise ValueError("Карты границ не должны быть пустыми")
    for edges in (pred, gt):
        if not np.all((edges == 0) | (edges == 1) | (edges == 255)):
            raise ValueError("Карты должны быть бинарными: 0 и 1 или 255")

    pred_objects = _closed_objects(pred)
    gt_objects = _closed_objects(gt)
    pred_areas = np.bincount(pred_objects.ravel())
    gt_areas = np.bincount(gt_objects.ravel())
    pred_count = len(pred_areas) - 1
    gt_count = len(gt_areas) - 1
    if pred_count + gt_count == 0:
        return 1.0

    overlap = (pred_objects > 0) & (gt_objects > 0)
    # Только встречающиеся пары: без плотной матрицы N_pred × N_gt.
    stride = gt_count + 1
    pair_ids, intersections = np.unique(
        pred_objects[overlap].astype(np.int64) * stride + gt_objects[overlap],
        return_counts=True,
    )
    pred_ids, gt_ids = pair_ids // stride, pair_ids % stride
    unions = pred_areas[pred_ids] + gt_areas[gt_ids] - intersections
    ious = intersections / unions
    # IoU > 0.5 даёт однозначное соответствие непересекающихся объектов.
    # TP + FP/2 + FN/2 = (N_pred + N_gt)/2.
    return float(ious[ious > 0.5].sum() / (0.5 * (pred_count + gt_count)))
