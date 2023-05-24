# Usage for model inference with onnxruntime

Required:

onnxruntime-1.7.0

opencv>=3.4.1

THe features (1x512) from a face images are extracted with ```model.onnx```.

## Install or download Onnxruntime library

search [https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases) to find a comfortable release. 

```shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.7.0/onnxruntime-linux-x64-1.7.0.tgz
tar -xvf onnxruntime-linux-x64-1.7.0.tgz
```

## Build 

```shell
mkdir build && cd build 
cmake .. -DONNXRUNTIME_DIR=./onnxruntime-linux-x64-1.7.0  -DCMAKE_BUILD_TYPE=Release
cmake --build .
// [Optinal if you not change this repo]
cp test ../ && cd ..
./test
```

## Reference
[1] https://github.com/microsoft/onnxruntime
