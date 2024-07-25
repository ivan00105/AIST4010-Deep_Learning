import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

import os
import time
import datetime

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import math
import torch.backends.cudnn as cudnn

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import argparse


torch.manual_seed(1234)


parser = argparse.ArgumentParser(description='Testing on multi-GPUs training with EfficientNet model')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--batch_size', type=int, default=512, help='')

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

"""The model architecture can be found in 
https://github.com/Wulingtian/EfficientNetv2_TensorRT_int8/blob/master/effnetv2.py
"""
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()



def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


class FaceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images, self.labels = self.load_images_labels()

    def load_images_labels(self):
        images = []
        labels = []
        for label_folder in os.listdir(self.folder_path):
            label_folder_path = os.path.join(self.folder_path, label_folder)
            if os.path.isdir(label_folder_path):
                for image_filename in os.listdir(label_folder_path):
                    image_path = os.path.join(label_folder_path, image_filename)
                    images.append(image_path)
                    labels.append(int(label_folder.split('_')[1]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
    
    # Initialize distributed environment
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('==> Making model..')
    model = effnetv2_s()
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    for param in model.parameters():
        param.requires_grad = False

    modules = list(model.modules())[::-1]
    for module in modules[:20]: # fine tune top 20 layers
        if not isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True

    
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    print(f'GPU {args.gpu} with batch size {args.batch_size} and {args.num_workers} workers')
    


    print('==> Preparing data..')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # transforms.GaussianBlur(21,10),
        # transforms.RandomEqualize(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        # transforms.RandomCrop(60, padding=2),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        # transforms.RandomErasing(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    def reduce_tensor(tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM) # sum loss & corrects across GPUs
        rt /= args.world_size  # average by the number of replicas
        return rt

    def train_model(model, train_dataset, val_dataset, criterion, optimizer, num_epochs, device, train_loader):
        # train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        best_acc = 0.0
        print("start training")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(device), labels.cuda(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # scheduler.step()
                
                # running_loss += loss.item() * inputs.size(0)
                running_loss += reduce_tensor(loss.data) * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                # running_corrects += torch.sum(preds == labels.data)
                running_corrects += reduce_tensor(torch.sum(preds == labels.data).float())

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)

            model.eval()
            val_loss = 0.0
            val_corrects = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(device), labels.cuda(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # val_loss += loss.item() * inputs.size(0)
                    val_loss += reduce_tensor(loss.data) * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    # val_corrects += torch.sum(preds == labels.data)
                    val_corrects += reduce_tensor(torch.sum(preds == labels.data).float())
                    
            val_loss = val_loss / len(val_dataset)
            val_acc = val_corrects.double() / len(val_dataset)
            if val_acc > best_acc and torch.distributed.get_rank() == 0:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'model_dist.pth')

            print(f'Epoch {epoch+1}/{num_epochs} on rank {torch.distributed.get_rank()}')
            print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            # if epoch % 2 == 1:
            scheduler.step(val_loss)

    train_data = FaceDataset("../data/train", transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=args.rank)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_data = FaceDataset("../data/val", transform=transform_test)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 350
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045, eps=1.0, alpha=0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.005)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.batch_size * (2**-14), alpha=1 - args.batch_size * (2**-14), eps=1e-3, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum = 0.9)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.97 ** (epoch / 2.4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True) 
    
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Model loaded from {args.resume}')
        else:
            print("no checkpoint found at '{}'".format(args.resume))
    
    train_model(model, train_data, val_data, criterion, optimizer, num_epochs, args.gpu, train_loader)
    

if __name__=='__main__':
    main()
    
    
