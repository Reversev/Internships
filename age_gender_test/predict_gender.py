import os
import glob
import torch
from PIL import Image
from torchvision import transforms
from mobilev3_gender import MobileFaceNet

classes_indict = ["female", "male"]
MODEL_PATH = './checkpoints/net_gender_best.pt'
TEST_PATH = './test/'
assert os.path.exists(TEST_PATH), "file: '{}' does not exists.".format(TEST_PATH)
device = torch.device('cpu')
print('Use device: ', device)
# pre-processing
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
net = MobileFaceNet(num_classes=2)
net = net.to(device)

# load model weights
assert os.path.exists(MODEL_PATH), "file: '{}' does not exist.".format(MODEL_PATH)
net.load_state_dict(torch.load(MODEL_PATH)["state_dict"])

for im_path in glob.glob(TEST_PATH + '*.jpg'):
    # load data
    im = Image.open(im_path)
    im = transform(im)  # [H, W, C] -> [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [C, H, W] -> [N, C, H, W]
    im = im.to(device)

    net.eval()
    with torch.no_grad():
        output = net(im)
        print(output)
        # confidence = torch.max(output, dim=1)[0].cpu().data.numpy()[0]  # option
        confidence = torch.max(torch.softmax(output, dim=1)).cpu().data.numpy()
        predict = torch.max(output, dim=1)[1].cpu().data.numpy()

    print('Vertification picture:', im_path.split('/')[-1], '\t',
          'Recognition result:', classes_indict[int(predict)], '\t',
          'Recognition confidence:', str(confidence))