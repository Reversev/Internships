import torch
from torch import nn
import math
import time 
import os
import argparse
from torchvision import transforms, datasets
import torch.optim as optim
import tqdm

 
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
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(axis=0, keepdim=True)
        res.append(correct_k.mul(100. / batch_size))
    return res
    

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
        self.linear7 = ConvBlock(256, 256, 4, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(256, feature_dim, 1, 1, 0, linear=True)
        self.classifer = nn.Linear(feature_dim, num_classes)

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
        x = self.classifer(x)
        # print(x.shape)
        return x


def train(net, train_loader, criterion, optimizer, device):
    net.train()
    losses = AverageMeter()
    train_correct1 = AverageMeter()
    for _, (train_imgs, train_labels) in enumerate(train_loader):
        # print(train_imgs.shape, train_labels.shape) # [64, 3, 64, 64] [64]
        # print(train_labels)
        train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)

        outputs = net(train_imgs)
        # print(outputs.shape)
        loss = criterion(outputs, train_labels)
        losses.update(loss.item(), train_labels.size(0))
        
        # calculate running acerage of accuracy 
        train_top1 = accuracy(outputs, train_labels, topk=(1, ))
        train_correct1.update(train_top1[0], train_labels.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    return losses.avg, train_correct1.avg


def valiadte(net, eval_loader, criterion, device):
    net.eval()
    top1 = AverageMeter()
    losses = AverageMeter()
    for _, (eval_imgs, eval_labels) in enumerate(eval_loader):
        eval_imgs, eval_labels = eval_imgs.to(device), eval_labels.to(device)
        output = net(eval_imgs)
        loss = criterion(output, eval_labels)
        pred1 = accuracy(output, eval_labels, topk=(1, ))
        losses.update(loss.item(), eval_imgs.size(0))
        top1.update(pred1[0], eval_imgs.size(0))
    return losses.avg, top1.avg


def loader(train_dir, eval_dir, args):
    data_transform = {"train": transforms.Compose([
                                transforms.RandomResizedCrop(64),
                                transforms.RandomHorizontalFlip(),
                                transforms.GaussianBlur((5, 5), sigma=(1.0, 1.0)),
                                transforms.RandomAffine(0, (0.1, 0.1)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                      "eval": transforms.Compose([
                              transforms.Resize((64, 64)),
                              transforms.GaussianBlur((5, 5), sigma=(1.0, 1.0)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True,
                                               drop_last=False)

    eval_dataset = datasets.ImageFolder(eval_dir, transform=data_transform["eval"])
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    return train_loader, eval_loader


def main(args):
    # test inference 
    x = torch.Tensor(2, 3, 64, 64)
    net = MobileFaceNet(num_classes=args.num_classes)
    s = time.time()
    y = net(x)
    e = time.time()
    print("inference time: ", str(e - s))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = os.path.join(args.img_path, args.dataset_name)
    assert os.path.join(image_path), "{} path does not exist!".format(image_path)
    # load loader 
    train_loader, eval_loader = loader(os.path.join(image_path, "train"), os.path.join(image_path, "eval"), args)

    # define optimizer and criterion
    optimizer = optim.SGD(net.parameters(), weight_decay=5e-4, lr=0.1, momentum=0.9, nesterov=True)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    best_acc1 = 0.0
    net = net.to(device)
    print("================Start================")
    st_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print("Train Epoch:{}/{} ...".format(epoch, args.epochs))

        train_loss, train_acc1 = train(net, train_loader, criterion, optimizer, device)

        eval_loss, eval_acc1 = valiadte(net, eval_loader, criterion, device)

        if best_acc1 < eval_acc1:
            best_acc1 = eval_acc1
            if not os.path.isdir("./checkpoints"):
                os.makedirs("./checkpoints")
            torch.save({'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict()}, 
                        './checkpoints/net_{}_base_best.pt'.format(args.dataset_name))
        else:
            pass

        row = {"Epoch": str(epoch),
               "Train_Loss": "%.3f" % train_loss, "Train_Acc@1": "%.3f" % train_acc1,
               "Test_Loss": "%.3f" % eval_loss, "Test_Acc@1": "%.3f" % eval_acc1,
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
    x = torch.Tensor(2, 3, 64, 64)
    net = MobileFaceNet(num_classes=args.num_classes)
    s = time.time()
    y = net(x)
    e = time.time()
    print("inference time: ", str(e - s))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mobileface for gender classifier")
    parser.add_argument('--dataset_name', '-d', default='gender')
    parser.add_argument('--img_path', '-p', default='./datasets/', help="images path")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training and eval(default: 128)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes (default: 2)')
    args = parser.parse_args()
    main(args)
    # test_demo()

