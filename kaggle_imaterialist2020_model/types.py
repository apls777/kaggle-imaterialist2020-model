from __future__ import annotations

from typing import NewType

import numpy as np
from typing_extensions import TypedDict


class Prediction(TypedDict):
    pred_num_detctions: int
    pred_image_info: np.array  # (2, 2)=(orginal|scale, height|width)  # noqa: E501
    pred_detection_boxes: np.array  # (num_detections, 4)
    pred_detection_classes: np.array  # (num_detections, )
    pred_detection_scores: np.array  # (num_detections, )
    pred_detection_masks: np.array  # (num_detections, mask_height, mask_width)


ImageHeight = NewType("Height", int)
ImageWidth = NewType("Width", int)
RLE = NewType("RLE", str)


class COCORLE(TypedDict):
    size: tuple[ImageHeight, ImageWidth]
    counts: RLE


BboxLeft = NewType("BboxLeft", float)
BboxTop = NewType("BboxTop", float)
BboxWidth = NewType("BboxWidth", float)
BboxHeight = NewType("BboxHeight", float)


class COCOAnnotation(TypedDict):
    image_id: int
    filename: str
    category_id: int
    # Avoid `bbox: list[float]` because
    # it's hard to know what each dimension means.
    # Also avoid `dict` like `{"left", "top", "width", "heiht"}`
    # along with the official COCO schema,
    # which adopts `list` instead of `dict`.
    bbox: tuple[BboxLeft, BboxTop, BboxWidth, BboxHeight]
    mask_area_fraction: float
    score: float
    segmentation: COCORLE
    mask_mean_score: float
