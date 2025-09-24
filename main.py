import cupy as cp
import cv2
import numpy as np
import tensorrt as trt

from yolo_utils import nms

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path: str):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


engine = load_engine("best.engine")
context = engine.create_execution_context()
INPUT_SIZE, MAX_VALUE_PIX = 640, 255

inputs, outputs, bindings = list(), list(), list()
for name in map(engine.get_tensor_name, range(engine.num_io_tensors)):
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    size = int(np.prod(shape))
    device_mem = cp.empty(size, dtype=dtype)
    bindings.append(int(device_mem.data.ptr))
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        inputs.append(device_mem)
    else:
        outputs.append(device_mem)


def infer(image: cv2.Mat):
    img_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img = img_resized.astype(np.float32) / MAX_VALUE_PIX
    img = np.transpose(img, (2, 0, 1))[np.newaxis]
    img = np.ascontiguousarray(img)
    cp.copyto(inputs[0], cp.asarray(img).ravel())
    context.execute_v2(bindings=bindings)
    return [out.get() for out in outputs]


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
        class_id = int(np.argmax(class_probs))
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
        tuple(int(c) for c in cv2.cvtColor(np.uint8([
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


if __name__ == "__main__":
    image_in = cv2.imread("test_0279.jpg")
    detections_in = infer(image_in)
    img_out = draw_boxes(image_in.copy(), detections_in, conf_threshold=.1, iou_threshold=.1)
    cv2.imwrite("detections_out.jpg", img_out)
    cv2.imshow("Detections", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
