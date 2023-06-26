# pedestrain attribute recongnition onnx cpp

## Usage
```shell
cmake .. -DONNXRUNTIME_DIR=/to/you/onnxruntime_path  -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4
./pede_recog required: <img path> <model path>
```

## Example:
```shell
cmake .. -DONNXRUNTIME_DIR=../onnxruntime-linux-x64-1.7.0  -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4
./pede_recog ../images/1.png ../onnx_models/ped_multilabel_uc_s1.onnx
```

|Datasets|Models|ma|Acc|Prec|Rec|F1|
|:------:|:---:|---|---|---|---|---|
| PA100k | ResNet18 | 79.22 | 77.76 | 87.11 | 85.92 | 86.51 |
| PA100k | resnet50 | 80.21 | 79.15 | 87.79 | 87.01 | 87.40 |

## Reference
[1] https://github.com/valencebond/Rethinking_of_PAR

