from typing import List

from numpy import argsort, array


def iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else .0


def nms(boxes: List[List[int]], scores: List[float], iou_threshold: float = .5) -> List[int]:
    idxs = argsort(scores)[::-1]
    keep = list()
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = array([iou(boxes[i], boxes[j]) for j in idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep
