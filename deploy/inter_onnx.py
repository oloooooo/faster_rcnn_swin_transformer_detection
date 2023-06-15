import numpy as np
import torchvision
import onnxruntime as ort
import cv2 as cv
from torchvision import transforms

transform = transforms.Compose([torchvision.transforms.ToTensor()])
sess_options = ort.SessionOptions()
# Below is for optimizing performance
sess_options.intra_op_num_threads = 24
# sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = ort.InferenceSession("faster_swin.onnx", sess_options=sess_options)
src = cv.imread("000012.jpg")
thr_score = 0.5

src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
src1 = np.expand_dims(src, axis=0)
src1 = np.transpose(src1, (0, 3, 1, 2)).astype(np.float32)

blob = transform(src)
c, h, w = blob.shape
input_x = blob.view(1, c, h, w)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


print(to_numpy(input_x))
print(src1)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_x)}
ort_outs = ort_session.run(None, ort_inputs)
boxes = ort_outs[0]  # boxes
labels = ort_outs[1]  # labels
scores = ort_outs[2]  # scores
print(boxes.shape, boxes.dtype, labels.shape, labels.dtype, scores.shape, scores.dtype)

index = 0
for x1, y1, x2, y2 in boxes:
    if scores[index] > thr_score:
        cv.rectangle(src, (np.int32(x1), np.int32(y1)),
                     (np.int32(x2), np.int32(y2)), (0, 255, 255), 1, 8, 0)
        label_id = labels[index]
        label_txt = str(label_id)
        cv.putText(src, label_txt, (np.int32(x1), np.int32(y1)), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
    index += 1
cv.imshow("Faster-RCNN Detection Demo", src)
cv.waitKey(0)
cv.destroyAllWindows()
