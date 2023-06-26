import argparse
import json
import os
import pickle

from dataset.augmentation import get_transform
from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from models.backbone import swin_transformer, resnet, bninception

set_seed(605)

clas_name = ['Hat','Glasses','ShortSleeve','LongSleeve','UpperStride','UpperLogo','UpperPlaid','UpperSplice',   # 8
             'LowerStripe','LowerPattern','LongCoat','Trousers','Shorts','Skirt&Dress','boots','HandBag','ShoulderBag',  # 9 
             'Backpack','HoldObjectsInFront','AgeOver60','Age18-60','AgeLess18','Female','Front','Side','Back']  # 9

def main(cfg, args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.NAME)

    train_tsfm, valid_tsfm = get_transform(cfg)
    # print(valid_tsfm)
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=26,
        c_in=c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    if args.checkpoint_path is not None:
        model = get_reload_weight(model_dir, model, pth=args.checkpoint_path) # 修改此处的模型名字
    else:
        assert "Please input a correct checkpoint path"

    model.eval()

    with torch.no_grad():
        for name in os.listdir(args.test_img):
            print(name)
            img = Image.open(os.path.join(args.test_img,name))
            img = valid_tsfm(img).cuda()
            img = img.view(1, *img.size())
            valid_logits, attns = model(img)

            valid_probs = torch.sigmoid(valid_logits[0]).cpu().numpy()
            print(valid_probs)
            valid_probs = valid_probs[0]>0.5

            res = []
            for i,val in enumerate(valid_probs):
                if val:
                    res.append(clas_name[i])
                if i ==14 and val==False:
                    res.append("female")
            print(res)
            print()


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_img", help="test images", type=str, default="./test_imgs",)
    parser.add_argument("--cfg", help="decide which cfg to use", type=str)
    parser.add_argument("--gpus", help="use gpus", type=str, default="0")
    parser.add_argument("--checkpoint_path", help="load checkpoint", type=str, default=None)
    parser.add_argument("--debug", type=str2bool, default="true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)
    main(cfg, args)

