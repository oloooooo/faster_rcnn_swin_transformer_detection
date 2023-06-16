# 这是一个简单的Faster rcnn主干网络替换为swin-transformer类的目标检测项目
##  环境搭建过程如下
    pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
    pip install pycocotools 
    pip install lxml
    pip install opencv-python=4.7.0
    pip install Pillow  
    如果不用部署下面的可以不装
    pip install onnxruntime 
    pip install onnx
    如果运行不了请参照我的环境python=3.8
    Package             Version
    ------------------- ------------
    certifi             2023.5.7    
    charset-normalizer  3.1.0       
    coloredlogs         15.0.1      
    contourpy           1.1.0
    cycler              0.10.0
    et-xmlfile          1.1.0
    filelock            3.9.0
    flatbuffers         23.5.26
    fonttools           4.40.0
    humanfriendly       10.0
    idna                3.4
    importlib-resources 5.12.0
    Jinja2              3.1.2
    joblib              1.0.1
    kiwisolver          1.3.1
    lxml                4.9.2
    MarkupSafe          2.1.2
    matplotlib          3.7.1
    mpmath              1.2.1
    networkx            3.0
    numpy               1.24.3
    onnx                1.14.0
    onnxruntime         1.15.0
    opencv-python       4.7.0.72
    openpyxl            3.0.7
    packaging           23.1
    Pillow              9.5.0
    pip                 23.1.2
    protobuf            4.23.3
    pycocotools         2.0.6
    pyparsing           2.4.7
    pyreadline3         3.4.1
    python-dateutil     2.8.2
    requests            2.31.0
    scikit-learn        0.24.1
    setuptools          67.8.0
    six                 1.15.0
    sklearn             0.0
    sympy               1.11.1
    threadpoolctl       2.1.0
    torch               2.0.0+cu117
    torchaudio          2.0.1+cu117
    torchvision         0.15.1+cu117
    typing_extensions   4.6.3
    urllib3             2.0.3
    wheel               0.38.4
    xlrd                2.0.1
    zipp                3.15.0

## 训练过程如下
请修改train.py下的这块代码已训练自己的VOC格式的数据集

    VOC_root = "dataset/"
    batch_size = 1
    nw = 1
    num_epochs = 15
值得注意的是dataset目录下应该是VOCdevkit文件夹、test.py文件是用来测试数据集格式的文件，你可以直接运行它来实现可视化数据集
最后修改pascal_voc_classes.json下直接的类别即可，注意你的下标必须从1开始。这也以为着在train.py的下列的numclass必须是你的类别数加1
最后你可以运行代码进行训练了。

## 测试
    label_json_path = './pascal_voc_classes.json'
    # load image
    original_img = Image.open("crazing_1.jpg")

是需要修改的部分。
### 部署过程请自己看程序
# 最后请注意
    请不要用torchvision的BackboneWithFPN因为torchvision的swintransformer通道对不上适用于图像分类的，我对其进行了部分修改
# 代码参考：
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing  
https://github.com/pytorch/vision
如果有什么问题可以联系luoxin982@qq.com
make by 罗鑫
