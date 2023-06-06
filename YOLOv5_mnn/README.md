# YOLOv5 MNN框架C++推理

MNN是阿里提出的深度网络加速框架，是一个轻量级的深度神经网络引擎，集成了大量的优化算子，支持深度学习的推理与训练。据说比腾讯开发的NCNN框架好一些。本文主要使用MNN对yolov5s模型进行推理加速。

## 查看Opencv版本（需要用到，建议提前安装）:
```bash
pkg-config --modversion opencv
```

## 编译MNN
```bash
git clone https://github.com/alibaba/MNN.git
cd MNN-master
mkdir build && cd build && cmake .. -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_QUANTOOLS=ON && make -
```
- 输出的结果会提示本地的硬件是否支持量化、数据处理加速等。

## 模型转化
本文是把官方的pt文件通过export.py转换成了onnx的文件格式，以后再运行如下命令：
```bash
./MNNConvert -f ONNX --modelFile ../models/yolov5s.onnx --MNNModel ../models/yolov5s.mnn --bizCode MNN
```
```yolov5s.mnn模型被存放在build同等目录下的models文件中（其他导入的文件亦是），需要存放于其他路径的话可以自行更改。```

mnn模型量化（如果支持的话）：
```bash
./quantized.out ../models/yolov5s.mnn ../models/yolov5s_quantized.mnn ../models/yolov5s_imageinputconfig.json
```

## 输出模型json文件
```bash
./MNNDump2Json ../models/yolov5s.mnn ../models/yolov5s_output.json
```


## 测试模型（10次）
```bash
./MNNV2Basic.out ../models/yolov5s.mnn 10 0 0 4 1x3x640x640 
```

- 第一个参数指定 待测试模型的二进制文件；
- 第二个参数指定 性能测试的循环次数，10就表示循环推理10次；
- 第三个参数指定 是否输出推理中间结果，0为不输出，1为只输出每个算子的输出结果（{op_name}.txt），2为输出每个算子的输入（Input_{op_name}.txt）和输出（{op_name}.txt）结果； 默认输出当前目录的output目录下（使用工具之前要自己建好output目录）；
- 第四个参数指定 执行推理的计算设备，有效值为 0（CPU）、1（Metal）、2（CUDA）、3（OpenCL）、6（OpenGL），7(Vulkan) ，9 (TensorRT)；
- 第五个参数为线程数，默认为4，仅对CPU有效；
- 第六个参数指定 输入tensor的大小，一般不需要指定。

## 测试模型速度（10次）
```bash
 ./timeProfile.out ../models/yolov5s.mnn 10 0 1x3x640x640 4
```

- 第一个参数 指定模型文件名；
- 第二个参数 指定运行次数，默认 100；
- 第三个参数 指定 执行推理的计算设备，有效值为 0（浮点 CPU）、1（Metal）、3（浮点OpenCL）、6（OpenGL），7(Vulkan)。（当执行推理的计算设备不为 CPU 时，Op 平均耗时和耗时占比可能不准）；
- 第四个参数 指定输入大小，可不设；
- 第五个参数 指定线程数，可不设，默认为 4。


## 将编译好的MNN库文件（include中的MNN）复制到[github项目](https://github.com/Reversev/Internships/tree/main/YOLOv5_mnn)
对应的include路径下
```bash
include
|-- MNN    ## 拷贝MNN-master下的文件夹
|   |-- AutoTime.hpp
|   |-- ErrorCode.hpp
|   |-- HalideRuntime.h
|   |-- ImageProcess.hpp
|   |-- Interpreter.hpp
|   |-- MNNDefine.h
|   |-- MNNForwardType.h
|   |-- MNNSharedContext.h
|   |-- Matrix.h
|   |-- Rect.h
|   |-- Tensor.hpp
|   |-- expr
|   |   |-- Executor.hpp
|   |   |-- ExecutorScope.hpp
|   |   |-- Expr.hpp
|   |   |-- ExprCreator.hpp
|   |   |-- MathOp.hpp
|   |   |-- Module.hpp
|   |   |-- NeuralNetWorkOp.hpp
|   |   |-- Optimizer.hpp
|   |   `-- Scope.hpp
|   |-- libMNN.so
|   `-- plugin
|       |-- PluginContext.hpp
|       |-- PluginKernel.hpp
|       `-- PluginShapeInference.hpp
|-- Yolo.h
|-- classnames.hpp
|-- time_tools.hpp
`-- util.h

3 directories, 28 files
```

具体命令如下（假设当前在build文件夹下）:
```shell
cp -r ../include/MNN <to_your-yolo-path>
cp libMNN.so <to_your-yolo-path>
```

## 运行C++文件推理（需要安装Opencv）
```bash
git clone https://github.com/Reversev/Internships/tree/main/YOLOv5_mnn.git
cd yolov5_mnn-master
mkdir build && cd build 
cmake ..
make install
cd ..
./yolov5-mnn "./checkpoints/yolov5s.mnn" "images/bus.jpg"
```
- ```./checkpoints/yolov5s.mnn```代表MNN模型权重的路径，默认设置为```./checkpoints/yolov5s.mnn```；
- ```images/bus.jpg```代表待推理的图片，默认设置为```images/000070.jpg```。

在工程文件中的src目录下的main.cpp(line 82-87)中，
```cpp
int num_classes=80;       // 模型推理类别数
std::vector<YoloLayerData> yolov5s_layers{
        {"482",    32, {{116, 90}, {156, 198}, {373, 326}}},  // 第一层的输出
        {"416",    16, {{30,  61}, {62,  45},  {59,  119}}},  // 第二层的输出
        {"output", 8,  {{10,  13}, {16,  30},  {33,  23}}},   // 第三层的输出
};
```
yolov5层模型的输出位置需要根据具体的模型进行设置，具体需要看MNN导出模型输出的相关层结构信息，如项目中所使用的5s模型，输出位置分别为```"482" and "416" and "output"```，其他以640分辨率作为模型输入的分辨率则需要改动这三个值，如果是使用1280分辨率（yolov5s6）的话需要多增加一层，并且修改先验框的数值，当然76行的```int INPUT_SIZE = 640;```也需要修改

如果需要修改相关的置信度信息，见main.cpp中的155-156行：
```cpp
float threshold = 0.25;     // classification confience threshold
float nms_threshold = 0.5;  // nms confience threshold
```

除了模型推理加速，MNN还提供了一些CV的API，让C++编写的图像处理代码更高效。

## 欢迎点赞收藏本文的项目🌟🌟🌟🚀🚀🚀

# Reference
[1] https://www.yuque.com/mnn/cn/wwueoh

[2] https://github.com/tymanman/yolov5_mnn

[3] MNN: A Universal and Efficient Inference Engine
