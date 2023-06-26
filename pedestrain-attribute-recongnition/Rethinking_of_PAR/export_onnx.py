import os
import cv2
import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify
import numpy as np
from models.model_factory import build_backbone, build_classifier
from models.base_block import FeatClassifier
from tools.function import get_model_log_path, get_reload_weight
from collections import OrderedDict
from models.backbone import swin_transformer, resnet, bninception


if __name__ == "__main__":
    height, width = 256, 192
    model_file = "ckpt_max_2023-06-21_18:53:44.pth"
    onnx_path =  "ped_multilabel_uc_s1.onnx"
    sim_onnx_path =  "ped_multilabel_uc_sim_s1.onnx"
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("source: ", model_file)
    print("result: ", onnx_path)
    exp_dir = os.path.join('exp_result', "PA100k")
    model_dir, log_dir = get_model_log_path(exp_dir, 'resnet18.base.adam')
    backbone, c_output = build_backbone("resnet18")
    classifier = build_classifier("linear")(
        nattr=26,
        c_in=c_output,
        bn=False,
        pool="avg",
        scale=1
    )
    inference_model = FeatClassifier(backbone, classifier)

    if model_file is not None:
        model_path = os.path.join(model_dir, model_file)
        print(model_path)
        load_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

        if isinstance(load_dict, OrderedDict):
            pretrain_dict = load_dict
        else:
            pretrain_dict = load_dict['state_dicts']
            print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")
    else:
        assert "Please input a correct checkpoint path"
    
    new_state_dict = OrderedDict()
    for k, v in pretrain_dict.items():
        name = k[7:]  # remove "module."
        new_state_dict[name] = v

    inference_model = inference_model.to(device)
    inference_model.load_state_dict(new_state_dict)
    inference_model.eval()

    # convert
    x = torch.randn(1, 3, height, width, requires_grad=False).to(device)  # moving the tensor to cpu
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
    image_path_ = "../attr_recog/images/099978.jpg"
    image_ = cv2.imread(image_path_)
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    image_ = cv2.resize(image_, (width, height))
    image_ = image_.astype(np.float32) / 255.0
    image_[:, :, 0] = (image_[:, :, 0] - 0.485) / 0.229
    image_[:, :, 1] = (image_[:, :, 1] - 0.456) / 0.224
    image_[:, :, 2] = (image_[:, :, 2] - 0.406) / 0.225
    image_ = image_.transpose((2, 0, 1))
    image_ = np.expand_dims(image_, 0)
    output = ort_session.run(['output'], input_feed={"input": image_})
    print("output: ", output)
