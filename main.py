import cupy as cp
import cv2
import numpy as np
import tensorrt as trt

from yolo_utils import draw_boxes

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


if __name__ == "__main__":
    image_in = cv2.imread("test_0279.jpg")
    detections_in = infer(image_in)
    img_out = draw_boxes(image_in.copy(), detections_in, conf_threshold=.1, iou_threshold=.1)
    cv2.imwrite("detections_out.jpg", img_out)
    cv2.imshow("Detections", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
