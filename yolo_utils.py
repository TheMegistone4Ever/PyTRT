from typing import List

import cv2
from numpy import argsort, argmax, array, uint8

from config import INPUT_SIZE, MAX_VALUE_PIX


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


def draw_boxes(
        image: cv2.Mat,
        detections,
        conf_threshold: float = .1,
        iou_threshold: float = .1,
        num_classes: int = 2,
        target_size: int = 1280,
):
    image = cv2.resize(image, (target_size, target_size))
    results = detections[0].reshape(-1, 7)
    boxes, scores, class_ids = list(), list(), list()

    for (cx, cy, bw, bh, conf_obj, *class_probs) in results:
        class_id = int(argmax(class_probs))
        score = conf_obj * class_probs[class_id]
        if score < conf_threshold:
            continue

        x1 = int((cx - bw / 2) * target_size / INPUT_SIZE)
        y1 = int((cy - bh / 2) * target_size / INPUT_SIZE)
        x2 = int((cx + bw / 2) * target_size / INPUT_SIZE)
        y2 = int((cy + bh / 2) * target_size / INPUT_SIZE)
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(class_id)

    keep_idx = nms(boxes, scores, iou_threshold)
    colors = [
        tuple(int(c) for c in cv2.cvtColor(uint8([
            [[c / num_classes * 179, MAX_VALUE_PIX, MAX_VALUE_PIX]]]), cv2.COLOR_HSV2BGR)[0][0])
        for c in range(num_classes)
    ]

    for (box, class_id, score) in map(lambda i: (boxes[i], class_ids[i], scores[i]), keep_idx):
        x1, y1, x2, y2 = box
        color = colors[class_id % num_classes]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{class_id}:{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return image
