import time

import torch
from PIL import Image
from torchvision import transforms

from train import create_model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=7)

    # load train weights
    model = torch.load("../save_weights/swin-model-63.pth")
    # load image
    original_img = Image.open("../dataset/VOCdevkit/VOC2007/JPEGImages/crazing_1.jpg")
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)
    # img = img.half()
    input_names = ['input']
    output_names = ['output']
    # model.half()
    model.eval()
    torch.onnx.export(model, img, 'faster_swin.onnx', input_names=input_names, output_names=output_names,
                      verbose=False,
                      opset_version=11, dynamic_axes={'input': {1: 'height', 2: 'width'}})


if __name__ == '__main__':
    main()
