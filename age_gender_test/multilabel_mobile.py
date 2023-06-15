import torch
from torch import nn
import math
import time 
import os
import copy
import argparse
from PIL import Image
from torchvision import transforms
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F

# age_label_ref = [[0, 12], [13, 23], [24, 29], [30, 34], [35, 39], [40, 49], [50, 59], [60, 80]]  # split classes from age 
age_label_ref = []
for i in range(2, 81, 1):
    age_label_ref.append([i, i])

pre_enconde = [0, ] * len(age_label_ref)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 计算准确率——方式1
# 设定一个阈值，当预测的概率值大于这个阈值，则认为这幅图像中含有这类标签\
def calculate_acuracy_mode_one(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num
 
    return precision, recall

# 计算准确率——方式2
# 取预测概率最大的前top个标签，作为模型的预测结果
def calculate_acuracy_mode_two(model_pred, labels):
    # 取前top个预测结果作为模型的预测结果
    precision = 0
    recall = 0
    top = 5
    # 对预测结果进行按概率值进行降序排列，取概率最大的top个结果作为模型的预测结果
    pred_label_locate = torch.argsort(model_pred, descending=True)[:, 0:top]
    for i in range(model_pred.shape[0]):
        temp_label = torch.zeros(1, model_pred.shape[1])
        temp_label[0,pred_label_locate[i]] = 1
        target_one_num = torch.sum(labels[i])
        true_predict_num = torch.sum(temp_label * labels[i])
        # 对每一幅图像进行预测准确率的计算
        precision += true_predict_num / top
        # 对每一幅图像进行预测查全率的计算
        recall += true_predict_num / target_one_num
    return precision, recall
    

class Multi_dataset(data.Dataset):
    def __init__(self, root, transform, target_transform=None, train: bool = True):
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.train = train
        self.imgs_list = []
        self.imgs_path = os.path.join(root, "original images")    # uncroped images
        self.split_txt_path = os.path.join(root, "image sets")
        self.txt_file_path = os.path.join(self.split_txt_path, "train.txt" if self.train else "val.txt")
        # self.f_img_classes_num = [0, ] * 81
        # self.m_img_classes_num = [0, ] * 81
        self.read_txt()
        # print(self.f_img_classes_num)
        # print(self.m_img_classes_num)
    
    def read_txt(self):
        with open(self.txt_file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            info = line.strip().split(" ")
            age = info[0].split("A")[1][:-4]
            ind = int(info[1])
            # if ind == 0:
            #     self.f_img_classes_num[int(age)] += 1 
            # else:
            #     self.m_img_classes_num[int(age)] += 1
            for key, ref in enumerate(age_label_ref):
                if(int(age)>=ref[0] and int(age)<=ref[1]):
                    if ind == 1: one_gender_hot = [1, 0]     # male
                    else: one_gender_hot = [0, 1]            # female
                    add_age_code = copy.copy(pre_enconde)
                    add_age_code[key] = 1
                    one_gender_hot.extend(add_age_code)
                    self.imgs_list.append([info[0], one_gender_hot])
                    # print(info[0], ind, key, one_gender_hot)


    def __getitem__(self, index):
        img_name, label = self.imgs_list[index]
        img_path = os.path.join(self.imgs_path, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.FloatTensor(label)   # for BCE loss
        return img, label
    

    def __len__(self):
        return len(self.imgs_list)


MobileFaceNet_BottleNeck_Setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]    


class BottleNeck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(BottleNeck, self).__init__()
        self.connect = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # 1*1 conv
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # 3*3 depth wise conv
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # 1*1 conv
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)

        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class MobileFaceNet(nn.Module):
    def __init__(self, num_classes=1000, feature_dim=128, bottleneck_setting=MobileFaceNet_BottleNeck_Setting):
        super(MobileFaceNet, self).__init__()
        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)
        self.num_classes = num_classes

        self.cur_channel = 64
        block = BottleNeck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 256, 1, 1, 0)
        self.linear7 = ConvBlock(256, 256, 7, 1, 0, dw=True, linear=True)   # 64: 4
        self.linear1 = ConvBlock(256, feature_dim, 1, 1, 0, linear=True)
        self.classifier = nn.Linear(feature_dim, 2 + num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.cur_channel, c, s, t))
                else:
                    layers.append(block(self.cur_channel, c, 1, t))
                self.cur_channel = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x



class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        # self.xs_pos = torch.sigmoid(x)
        self.xs_pos = x
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


def train(net, train_loader, criterion, optimizer, device):
    net.train()
    losses = AverageMeter()
    train_precision = AverageMeter()
    train_recall = AverageMeter()
    for _, (train_imgs, train_labels) in enumerate(train_loader):
        # print(train_imgs.shape, train_labels.shape) # [64, 3, 112, 112] [112, 10]
        # print(train_labels)
        train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)

        outputs = net(train_imgs)
        # print(outputs.shape)
        loss = criterion(outputs, train_labels)
        losses.update(loss.item(), train_labels.size(0))
        
        # calculate running acerage of accuracy 
        precision, recall = calculate_acuracy_mode_one(outputs, train_labels)
        train_precision.update(precision.item(), train_labels.size(0))
        train_recall.update(recall.item(), train_labels.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, train_precision.avg, train_recall.avg


def valiadte(net, eval_loader, criterion, device):
    net.eval()
    eval_precision = AverageMeter()
    eval_recall = AverageMeter()
    losses = AverageMeter()
    for _, (eval_imgs, eval_labels) in enumerate(eval_loader):
        eval_imgs, eval_labels = eval_imgs.to(device), eval_labels.to(device)
        output = net(eval_imgs)
        loss = criterion(output, eval_labels)
        precision, recall = calculate_acuracy_mode_one(output, eval_labels)
        losses.update(loss.item(), eval_imgs.size(0))
        eval_precision.update(precision.item(), eval_imgs.size(0))
        eval_recall.update(recall.item(), eval_imgs.size(0))

    return losses.avg, eval_precision.avg, eval_recall.avg


def multi_loader(img_dir, args):
    data_transform = {"train": transforms.Compose([
                                transforms.RandomResizedCrop(112),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                      "eval": transforms.Compose([
                              transforms.Resize((112, 112)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    train_dataset = Multi_dataset(root=img_dir, transform=data_transform["train"], train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=3, pin_memory=True,
                                               drop_last=False)

    eval_dataset = Multi_dataset(root=img_dir, transform=data_transform["eval"], train=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=3, pin_memory=True, drop_last=False)
    
    return train_loader, eval_loader


def main(args):
    # test inference 
    x = torch.Tensor(2, 3, 112, 112)
    net = MobileFaceNet(num_classes=len(age_label_ref))
    s = time.time()
    y = net(x)
    e = time.time()
    print("the number of classes: ", len(age_label_ref), ", the inference time: ", str(e - s))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = os.path.join(args.img_path, args.dataset_name)
    assert os.path.join(image_path), "{} path does not exist!".format(image_path)
    # load loader 
    train_loader, eval_loader = multi_loader(image_path, args)

    # define optimizer and criterion
    # optimizer = optim.SGD(net.parameters(), weight_decay=5e-4, lr=0.1, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-4, lr=0.001)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, ], gamma=0.1)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0001)
    criterion = nn.BCELoss().to(device)
    # criterion = AsymmetricLossOptimized(gamma_neg=1, gamma_pos=0, clip=0.01, disable_torch_grad_focal_loss=True)

    best_acc1 = 0.0
    net = net.to(device)
    print("================Start================")
    st_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print("Train Epoch:{}/{} ...".format(epoch, args.epochs))

        train_loss, train_precision, train_recall = train(net, train_loader, criterion, optimizer, device)

        eval_loss, eval_precision, eval_recall = valiadte(net, eval_loader, criterion, device)

        if best_acc1 < eval_precision:
            best_acc1 = eval_precision
            if not os.path.isdir("./checkpoints"):
                os.makedirs("./checkpoints")
            torch.save({'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict()}, 
                        './checkpoints/net_{}_multilabel_uc_best_s.pt'.format(args.dataset_name))
        else:
            pass

        row = {"Epoch": str(epoch),
               "Train_Loss": "%.3f" % train_loss, "Train_P@1": "%.3f" % (100. * train_precision), "Train_R@1": "%.3f" % (100. * train_recall),
               "Test_Loss": "%.3f" % eval_loss, "Test_P@1": "%.3f" % (100. * eval_precision), "Test_R@1": "%.3f" % (100. * eval_recall),
               "Test_best_Acc@1": "%.3f" % (100. * best_acc1), "lr": "%.4f" % lr_scheduler.get_last_lr()[0]}
        row.update({
            'time': int(time.time() - st_time),
            'eta': int(time.time() - st_time) / (epoch + 1) * (args.epochs - epoch - 1),
        })
        print(row)    
        lr_scheduler.step()

    print("=> Training Finish!")


def test_demo():
    # test inference 
    x = torch.Tensor(2, 3, 112, 112)
    net = MobileFaceNet(num_classes=len(age_label_ref))
    s = time.time()
    y = net(x)
    e = time.time()
    print("inference time: ", str(e - s))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mobileface for gender classifier")
    parser.add_argument('--dataset_name', '-d', default='All-Age-Faces-Dataset')
    parser.add_argument('--img_path', '-p', default='./datasets/', help="images path")
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training and eval(default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    args = parser.parse_args()
    main(args)
    # test_demo()

