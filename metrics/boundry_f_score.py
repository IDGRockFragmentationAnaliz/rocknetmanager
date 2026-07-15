import numpy as np
import cv2


def boundary_f_score(
    edges_gt: np.ndarray,
    edges_pred: np.ndarray,
    tolerance_px: int = 3,
) -> float:
    """
    F-score для двух однопиксельных бинарных карт границ.

    Ожидаемый формат:
        dtype: uint8
        граница: 255
        фон: 0

    tolerance_px:
        Допустимое отклонение между линиями в пикселях.
    """
    if edges_gt.ndim != 2 or edges_pred.ndim != 2:
        raise ValueError("Обе карты должны быть двумерными")

    if edges_gt.shape != edges_pred.shape:
        raise ValueError(
            f"Размеры не совпадают: "
            f"GT={edges_gt.shape}, prediction={edges_pred.shape}"
        )

    if edges_gt.dtype != np.uint8 or edges_pred.dtype != np.uint8:
        raise TypeError("Обе карты должны иметь dtype=np.uint8")

    if tolerance_px < 0:
        raise ValueError("tolerance_px должен быть >= 0")

    gt = edges_gt == 255
    pred = edges_pred == 255

    gt_count = int(np.count_nonzero(gt))
    pred_count = int(np.count_nonzero(pred))

    if gt_count == 0 and pred_count == 0:
        return 1.0

    if tolerance_px == 0:
        gt_dilated = gt
        pred_dilated = pred
    else:
        kernel_size = 2 * tolerance_px + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

        gt_dilated = cv2.dilate(
            gt.astype(np.uint8),
            kernel,
        ).astype(bool)

        pred_dilated = cv2.dilate(
            pred.astype(np.uint8),
            kernel,
        ).astype(bool)

    # Precision:
    # доля предсказанных пикселей, попавших в допустимую зону GT.
    matched_pred = pred & gt_dilated
    matched_pred_count = int(np.count_nonzero(matched_pred))

    precision = (
        matched_pred_count / pred_count
        if pred_count > 0
        else 0.0
    )

    # Recall:
    # доля пикселей GT, рядом с которыми есть предсказанная граница.
    matched_gt = gt & pred_dilated
    matched_gt_count = int(np.count_nonzero(matched_gt))

    recall = (
        matched_gt_count / gt_count
        if gt_count > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return f1