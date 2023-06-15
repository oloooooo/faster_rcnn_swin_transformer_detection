import argparse
import os
import random
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# names = ["1", "2", "3", "4", "5", "6", "unknow" 'truck', 'boat', 'traffic light',
#          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#          'hair drier', 'toothbrush']

names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


class ONNX_engine():
    def __init__(self, weights, size, cuda) -> None:
        self.img_new_shape = (size, size)
        self.weights = weights
        self.device = cuda
        self.init_engine()
        self.names = names
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}

    def init_engine(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.weights, providers=providers)

    def predict(self, im):
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]: im}
        outputs = self.session.run(outname, inp)
        # print(outputs.shape)
        return outputs

    def preprocess(self, image_path):
        print('----------', image_path, '---------------')
        self.img = cv2.imread(image_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        image = self.img.copy()
        im, ratio, dwdh = self.letterbox(image, auto=False)
        t1 = time.time()
        outputs = self.predict(im)
        # print(outputs)
        print("推理时间", (time.time() - t1) * 1000, ' ms')
        ori_images = [self.img.copy()]
        # for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        #     image = ori_images[int(batch_id)]
        #     box = np.array([x0, y0, x1, y1])
        #     box -= np.array(dwdh * 2)
        #     box /= ratio
        #     box = box.round().astype(np.int32).tolist()
        #     cls_id = int(cls_id)
        #     score = round(float(score), 3)
        #     name = self.names[cls_id]
        #     color = self.colors[name]
        #     name += ' ' + str(score)
        #     print("pre result is :", box, name)
        #     print(type(box[:2]), box[2:])
        #     cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 2)
        #     cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=1)
        boxes = outputs[0]  # boxes
        labels = outputs[1]  # labels
        scores = outputs[2]  # scores
        # print(boxes.shape, boxes.dtype, labels.shape, labels.dtype, scores.shape, scores.dtype)

        index = 0
        for x1, y1, x2, y2 in boxes:
            label_id = labels[index]
            label_txt = names[label_id-1]
            if scores[index] > 0.5:
                cv2.rectangle(ori_images[0], (np.int32(x1), np.int32(y1)),
                              (np.int32(x2), np.int32(y2)), color=self.colors[label_txt], thickness=2, lineType=8, shift=0)

                cv2.putText(ori_images[0], label_txt, (np.int32(x1), np.int32(y1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0,0), 1)
            index += 1
        a = Image.fromarray(ori_images[0])
        return a

    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # 调整大小和垫图像，同时满足跨步多约束
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.img_new_shape

        # 如果是单个size的话，就在这里变成一双
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 尺度比 (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 只缩小，不扩大(为了更好的val mAP)
            r = min(r, 1.0)

        # 计算填充
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # 最小矩形区域
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255
        return im, r, (dw, dh)


if __name__ == '__main__':
    # # parser = argparse.ArgumentParser()
    # # parser.add_argument('--weights', type=str, default=r'best.onnx', help='weights path of onnx')
    # # parser.add_argument('--cuda', type=bool, default=False, help='if your pc have cuda')
    # # parser.add_argument('--imgs_path', type=str, default=r'inference/images', help='infer the img of path')
    # # parser.add_argument('--size', type=int, default=256, help='infer the img size')
    # # opt = parser.parse_args()
    # size = 200
    # weights = 'faster_swin_map0.761.onnx'
    # cuda = False
    # imgs_path = 'images'
    # save_path = 'onnx_interf'
    # onnx_engine = ONNX_engine(weights, size, True)
    #
    # for img_path in os.listdir(imgs_path):
    #     img_path_file = imgs_path + '/' + img_path
    #     # print('The img path is: ', img_path_file)
    #     a = onnx_engine.preprocess(img_path_file)
    #     a.save(save_path + '/' + img_path)
    #     print('*' * 50)
    size = 200
    weights = 'faster_swin.onnx'
    cuda = False
    img_path = '000012.jpg'
    save_path = 'onnx_interf'
    onnx_engine = ONNX_engine(weights, size, cuda)
    tmp="tmp.jpg"
    # print('The img path is: ', img_path_file)
    a = onnx_engine.preprocess(img_path)
    a.save(tmp)