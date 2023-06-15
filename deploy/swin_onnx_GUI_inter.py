# from mmdet.apis import init_detector, inference_detector
# import argparse
import os
import random
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, \
    QListWidgetItem, QLineEdit

pic_h = 50
names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


class ONNX_engine():
    def __init__(self, weights, size, cuda, thr) -> None:
        self.img_new_shape = (size, size)
        self.weights = weights
        self.device = cuda
        self.init_engine()
        self.names = names
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}
        self.thr = thr

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
            label_txt = names[label_id - 1]
            if scores[index] > self.thr:
                cv2.rectangle(ori_images[0], (np.int32(x1), np.int32(y1)),
                              (np.int32(x2), np.int32(y2)), color=self.colors[label_txt], thickness=2, lineType=8,
                              shift=0)

                cv2.putText(ori_images[0], label_txt+str(round(scores[index],2)), (np.int32(x1), np.int32(y1)), cv2.FONT_HERSHEY_PLAIN, 1.0,
                            (0, 0, 0), 1)
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


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, 1024, 1000)
        self.image_files = []
        self.current_image_index = 0
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        # self.image_label.setFixedSize(00, 600)
        self.checkpoint_label = QLabel('检查点文件路径：')
        self.checkpoint_edit = QLineEdit(r"faster_swin_map0.761.onnx")
        # self.img_label = QLabel('图像路径：')
        self.device_label = QLabel('设备：')
        self.device_edit = QLineEdit('cpu')
        self.directory_label = QLabel("需要检测的图片文件夹路径")
        self.directory = QLineEdit(r"images")
        self.thr_label = QLabel('置信度')
        self.thr = QLineEdit('0.7')
        self.detect_button = QPushButton("Link start")
        self.detect_button.clicked.connect(self.load_image_files)
        self.previous_button = QPushButton('上一张')
        self.previous_button.clicked.connect(self.show_previous_image)
        self.result_label = QLabel()
        self.next_button = QPushButton('下一张')
        self.next_button.clicked.connect(self.show_next_image)
        # self.select_device_GPU = QPushButton('利用gpu检测')
        # self.select_device_GPU.clicked.connect(self.Gpu_Run)
        # self.select_device_CPU = QPushButton('利用cpu检测')
        # self.select_device_CPU.clicked.connect(self.Cpu_Run)
        # self.notice = QLabel("正在使用cpu检测")
        self.which_to_detect = QLabel()
        self.model = ONNX_engine(self.checkpoint_edit.text(), size=200, cuda=False, thr=float(self.thr.text()))
        # self.inference_button = QPushButton('推理')
        # self.inference_button.clicked.connect(self.run_inference)
        self.image_list = QListWidget()
        self.image_list.setFixedSize(QSize(1280, 200))
        self.image_list.itemSelectionChanged.connect(self.show_selected_image)
        self.image_list.setSelectionMode(QListWidget.SingleSelection)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.previous_button)
        button_layout.addWidget(self.next_button)
        device_layout = QHBoxLayout()
        # device_layout.addWidget(self.select_device_CPU)
        # device_layout.addWidget(self.select_device_GPU)
        # button_layout.addWidget(self.inference_button)
        pic_layout = QHBoxLayout()
        pic_layout.addWidget(self.image_label)
        pic_layout.addWidget(self.result_label)
        pic_layout.setSpacing(1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.detect_button)
        main_layout.addWidget(self.image_list)
        main_layout.addWidget(self.directory_label)
        main_layout.addWidget(self.directory)
        main_layout.addWidget(self.thr_label)
        main_layout.addWidget(self.thr)
        main_layout.addLayout(device_layout)
        main_layout.addLayout(pic_layout)
        main_layout.addLayout(button_layout)
        # main_layout.addWidget(self.notice)
        main_layout.addWidget(self.which_to_detect)
        self.setLayout(main_layout)

        # self.load_image_files()

        self.show_current_image()
        # self.populate_image_list()

    def load_image_files(self):
        directory = self.directory.text()
        print(directory)
        if directory:
            for filename in os.listdir(directory):
                if filename.endswith('.jpg'):
                    self.image_files.append(os.path.join(directory, filename))
        for image_file in self.image_files:
            item = QListWidgetItem(os.path.basename(image_file))
            item.setData(Qt.UserRole, image_file)
            self.image_list.addItem(item)

    def show_selected_image(self):
        selected_item = self.image_list.currentItem()
        if selected_item:
            image_path = selected_item.data(Qt.UserRole)
            # print(image_path)
            image = QImage(image_path)
            pixmap = QPixmap.fromImage(image).scaled(500, 500)
            # pixmap = pixmap.scaled(640, 640, Qt.SmoothTransformation)
            self.current_image_index = self.image_files.index(image_path)
            self.image_label.setPixmap(pixmap)
            self.run_inference(image_path)

    def show_current_image(self):
        if self.image_files:
            image_path = self.image_files[self.current_image_index]
            image = QImage(image_path)
            pixmap = QPixmap.fromImage(image)
            pixmap = pixmap.scaled(500, 500)
            self.image_label.setPixmap(pixmap)
            self.run_inference(image_path)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()
        else:
            self.current_image_index = len(self.image_files) - 1
            self.show_current_image()

    def show_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_current_image()
        else:
            self.current_image_index = 0
            self.show_current_image()

    def run_inference(self, image_path):
        # image_path = self.image_files[self.current_image_index]
        # print(image_path)
        # print("yes, you did it")
        # 从文本框中获取参数值
        # 初始化检测器模型
        img = image_path
        self.which_to_detect.setText(img)
        # 运行目标检测
        a = self.model.preprocess(image_path)
        a.save('detect.png')
        # self.model.show_result(img, result, out_file='detect.png', score_thr=float(self.thr.text()))
        # 在结果标签中显示检测结果
        self.result_label.setPixmap(QPixmap('detect.png').scaled(500, 500))

        # self.before_label.setPixmap(QPixmap(img))

    def Cpu_Run(self):
        self.notice.setText("切换至CPU")
        self.model = ONNX_engine(self.checkpoint_edit.text(), size=200, cuda=False, thr=float(self.thr.text()))

    def Gpu_Run(self):
        try:
            self.notice.setText("切换至GPU")
            self.model = ONNX_engine(self.checkpoint_edit.text(), size=200, cuda=False, thr=float(self.thr.text()))
        except:
            self.notice.setText("切换至GPU失败，正在使用cpu")
            self.model = ONNX_engine(self.checkpoint_edit.text(), size=200, cuda=False, thr=float(self.thr.text()))


if __name__ == '__main__':
    app = QApplication([])

    image_viewer = ImageViewer()
    image_viewer.setWindowTitle('detect for some  defects of steel')
    image_viewer.show()
    app.exec_()
