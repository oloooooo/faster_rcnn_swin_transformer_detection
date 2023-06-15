import datetime
import os

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

import transforms
from network_files import BackboneWithFPN
from engine import train_one_epoch, evaluate
from my_dataset import VOCDataSet
from swin_transformer import swin_t, Swin_T_Weights


def create_model(num_classes):
    backbone = swin_t(weights=Swin_T_Weights.DEFAULT).features

    # print(tmp)
    print([name for name, _ in backbone.named_children()])
    return_nodes = {'1': '0', '2': '1', '4': '2', '6': '3'}

    # backbone = create_feature_extractor(backbone, return_nodes)
    # print(backbone)
    # exit()
    backbone = BackboneWithFPN(backbone,
                               return_layers=return_nodes,
                               in_channels_list=[96, 96 * 2, 96 * 2 * 2, 96 * 2 * 2 * 2],
                               out_channels=256,
                               extra_blocks=LastLevelMaxPool())
    # print([name for name, _ in tmp.named_children()])
    # print(backbone_with_fpn)
    # print(backbone_with_fpn)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    model = FasterRCNN(backbone,
                       num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler, box_head=box_head, rpn_head=rpn_head
                       )

    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    VOC_root = "dataset/"  # VOCdevkit
    batch_size = 1
    nw = 1
    num_epochs = 15
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, "2007", data_transform["train"], "train.txt")

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, "2007", data_transform["val"], "val.txt")

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

    # # 获取第一个 batch 的数据和标签
    # batch_idx, (examples_data, examples_targets) = next(enumerate(train_data_loader))
    # # 获取第一张图像和对应的标签
    # img = examples_data[0][0]
    # targets = examples_targets[0]
    # # 绘制图像
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)
    # # 绘制目标边界框
    # for bbox in targets['boxes'].cpu().numpy():
    #     rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r',
    #                          facecolor='none')
    #     ax.add_patch(rect)
    # # 显示图像
    # plt.show()
    # exit()
    # create model num_classes equal background + your classes
    model = create_model(num_classes=7)
    # print(model)
    model.to(device)

    for param in model.backbone.parameters():
        param.requires_grad = True
    for param in model.roi_heads.parameters():
        param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    # print(len(params))
    # exit()
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=30,
                                                   gamma=0.33)
    # learning rate scheduler
    for epoch in range(0, num_epochs, 1):
        train_one_epoch(model, optimizer, train_data_loader,
                        device, epoch, print_freq=50,
                        scaler=None)
        lr_scheduler.step()
        evaluate(model, val_data_loader, device=device)
        if epoch % 3 == 0 or epoch == 0 or epoch == num_epochs - 1:
            torch.save(model, "save_weights/swin-model-{}.pth".format(epoch))



if __name__ == "__main__":
    main()
