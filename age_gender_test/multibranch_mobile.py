import torch
from torch import nn
import math
import time 
import os
import argparse
from PIL import Image
from torchvision import transforms
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F


age_label_ref = [[0, 12], [13, 23], [24, 29], [30, 34], [35, 39], [40, 49], [50, 59], [60, 80]]  # split classes from age 


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


@torch.no_grad()
def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)
    # print(output.shape)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    # print(pred.shape)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(axis=0, keepdim=True)
        res.append(correct_k.mul(100. / batch_size))
    return res
    

class Multi_dataset(data.Dataset):
    def __init__(self, root, transform, target_transform=None, train: bool = True):
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.train = train
        self.imgs_list = []
        self.imgs_path = os.path.join(root, "aglined faces") 
        self.split_txt_path = os.path.join(root, "image sets")
        self.txt_file_path = os.path.join(self.split_txt_path, "train.txt" if self.train else "val.txt")
        self.read_txt()
    
    def read_txt(self):
        with open(self.txt_file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            info = line.strip().split(" ")
            
            age = info[0].split("A")[1][:-4]
            for key, ref in enumerate(age_label_ref):
                if(int(age)>=ref[0] and int(age)<=ref[1]):
                    self.imgs_list.append([info[0], key, int(info[1])])  # [img_name, age classes, gender classes
                    # print(info[0], key, int(info[1]))


    def __getitem__(self, index):
        img_name, age_label, gender_label = self.imgs_list[index]
        img_path = os.path.join(self.imgs_path, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, age_label, gender_label
    

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
        self.age_classifer = nn.Linear(feature_dim, self.num_classes)
        self.gender_classifier = nn.Linear(feature_dim, 2)
        print("the number of age classifer: ", num_classes)

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
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.dw_conv1(x)
        # print(x.shape)
        x = self.blocks(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.linear7(x)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        a = self.age_classifer(x)
        # print(a.shape)
        g = torch.sigmoid(self.gender_classifier(x))
        # print(g.shape)
        return a, g 


def train(net, train_loader, criterion_age, criteron_gender, optimizer, device, args):
    net.train()
    losses = AverageMeter()
    train_age_correct1 = AverageMeter()
    train_gender_correct1 = AverageMeter()
    for _, (train_imgs, train_age_labels, train_gender_labels) in enumerate(train_loader):
        # train_gender_labels = F.one_hot(train_gender_labels).to(torch.float32)
        train_imgs, train_age_labels, train_gender_labels = train_imgs.to(device), train_age_labels.to(device), train_gender_labels.to(device)
        num = train_age_labels.size(0)
        age_outputs, gender_outputs = net(train_imgs)
        age_loss = criterion_age(age_outputs, train_age_labels)
        losses.update(age_loss.item(), num)
        # print(gender_outputs, train_gender_labels)
        gender_loss = criteron_gender(gender_outputs, train_gender_labels)
        losses.update(args.labma * gender_loss.item(), num)
        loss = (age_loss + args.labma * gender_loss) / 2
        
        # calculate running acerage of accuracy 
        train_age_top = accuracy(age_outputs, train_age_labels, topk=(1, ))
        train_age_correct1.update(train_age_top[0], num)
        train_gender_top = accuracy(gender_outputs, train_gender_labels, topk=(1, ))
        train_gender_correct1.update(train_gender_top[0], num)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, train_age_correct1.avg, train_gender_correct1.avg


def valiadte(net, eval_loader, criterion_age, criterion_gender, device):
    net.eval()
    age_top1 = AverageMeter()
    gender_top1 = AverageMeter()
    losses = AverageMeter()
    for _, (eval_imgs, eval_age_labels, eval_gender_labels) in enumerate(eval_loader):
        eval_imgs, eval_age_labels, eval_gender_labels = eval_imgs.to(device), eval_age_labels.to(device), eval_gender_labels.to(device)
        age_output, gender_output = net(eval_imgs)

        num = eval_age_labels.size(0)
        loss_age = criterion_age(age_output, eval_age_labels)
        age_pred = accuracy(age_output, eval_age_labels, topk=(1, ))
        losses.update(loss_age.item(), num)
        age_top1.update(age_pred[0], num)
    
        loss_gender = criterion_gender(gender_output, eval_gender_labels)
        gender_pred = accuracy(gender_output, eval_gender_labels, topk=(1, ))
        losses.update(loss_gender.item(), num)
        gender_top1.update(gender_pred[0], num)

    return losses.avg, age_top1.avg, gender_top1.avg


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
    print("inference time: ", str(e - s))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = os.path.join(args.img_path, args.dataset_name)
    assert os.path.join(image_path), "{} path does not exist!".format(image_path)

    # load loader 
    train_loader, eval_loader = multi_loader(image_path, args)

    # define optimizer and criterion
    optimizer = optim.SGD(net.parameters(), weight_decay=5e-4, lr=0.1, momentum=0.9, nesterov=True)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    criterion_age = nn.CrossEntropyLoss().to(device)
    criterion_gender = nn.CrossEntropyLoss().to(device)

    best_acc1 = 0.0
    net = net.to(device)
    print("================Start================")
    st_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print("Train Epoch:{}/{} ...".format(epoch, args.epochs))

        train_loss, train_age_acc, train_gender_acc = train(net, train_loader, criterion_age, criterion_gender, optimizer, device, args)

        eval_loss, eval_age_acc, eval_gender_acc = valiadte(net, eval_loader, criterion_age, criterion_gender, device)

        if best_acc1 < ((eval_gender_acc + eval_age_acc) / 2.):
            best_acc1 = (eval_gender_acc + eval_age_acc) / 2.
            if not os.path.isdir("./checkpoints"):
                os.makedirs("./checkpoints")
            torch.save({'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict()}, 
                        './checkpoints/net_{}_multibranch_best.pt'.format(args.dataset_name))
        else:
            pass

        # row = {"Epoch": str(epoch),
        #        "Train_Loss": "%.3f" % train_loss, "Train_A_Acc@1": "%.3f" % train_age_acc, "Train_G_Acc@1": "%.3f" % train_gender_acc,
        #        "Test_Loss": "%.3f" % eval_loss, "Test_A_Acc@1": "%.3f" % eval_age_acc, "Test_G_Acc@1": "%.3f" % eval_gender_acc,
        #        "Test_best_Acc@1": "%.3f" % best_acc1, "lr": "%.4f" % lr_scheduler.get_lr()[0]}
        row = {"Epoch": str(epoch),
               "Train_Loss": "%.3f" % train_loss, "Train_A_Acc@1": "%.3f" % train_age_acc, "Train_G_Acc@1": "%.3f" % train_gender_acc,
               "Test_Loss": "%.3f" % eval_loss, "Test_A_Acc@1": "%.3f" % eval_age_acc, "Test_G_Acc@1": "%.3f" % eval_gender_acc,
               "Test_best_Acc@1": "%.3f" % best_acc1, "lr": "%.4f" % lr_scheduler.get_last_lr()[0]}        
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
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes (default: 2)')
    parser.add_argument('--labma', type=float, default=0.5,
                        help='the balance parameter of gender classifier')
    args = parser.parse_args()
    main(args)
    # test_demo()

