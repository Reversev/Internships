import os
import cv2
import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify
import numpy as np
from multilabel_mobile import MobileFaceNet


if __name__ == "__main__":
    input_size = 112
    batch_size = 1
    num_classes = 79
    model_file = "./checkpoints/net_All-Age-Faces-Dataset_multilabel_uc_best_s.pt"
    onnx_path =  "./checkpoints/net_multilabel_uc_s.onnx"
    sim_onnx_path =  "./checkpoints/net_multilabel_uc_sim_s.onnx"
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("source: ", model_file)
    print("result: ", onnx_path)
    inference_model = MobileFaceNet(num_classes=num_classes)  # the number of age 
    loaded_model = torch.load(model_file, map_location=device)
    inference_model.load_state_dict(loaded_model['state_dict'])
    inference_model = inference_model.to(device)
    inference_model.eval()

    x = torch.randn(batch_size, 3, 112, 112, requires_grad=False).to(device)  # moving the tensor to cpu
    torch.onnx.export(inference_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output']  # the model's output names
                      )

    print("export onnx done.")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    model_simp, check = simplify(onnx_model, check_n=3)
    onnx.save(model_simp, sim_onnx_path)
    print("export onnx sim done.")

    # test onnxruntime
    ort_session = ort.InferenceSession(sim_onnx_path)

    image_path_ = "./test.jpg"
    image_ = cv2.imread(image_path_)  # keep BGR
    image_ = cv2.resize(image_, (input_size, input_size))
    image_ = image_.astype(np.float32) / 255.0
    image_[:, :, 0] = (image_[:, :, 0] - 0.485) / 0.229
    image_[:, :, 1] = (image_[:, :, 1] - 0.456) / 0.224
    image_[:, :, 2] = (image_[:, :, 2] - 0.406) / 0.225
    image_ = image_.transpose((2, 0, 1))  # (3,64,64)

    image_ = np.expand_dims(image_, 0)

    output = ort_session.run(['output'], input_feed={"input": image_})

    print("output: ", output)

    """
    PYTHONPATH=. python3 ./export_onnx.py
    """