import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import math

torch.manual_seed(1234)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.GaussianBlur(21,10),
    # transforms.RandomEqualize(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    # transforms.RandomCrop(60, padding=2),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    transforms.RandomErasing(),
])


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



train_data = FaceDataset("data/train", transform=transform)
val_data = FaceDataset("data/val", transform=transform)

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

     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = effnetv2_s()
# model.load_state_dict(torch.load('model_dist.pth')['model_state_dict'])

# model.to(device)


criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045, eps=1.0, alpha=0.9)
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
# optimizer.load_state_dict(torch.load('model2.pth')['optimizer_state_dict'])
# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
# scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.005)

def train_model(model, train_dataset, val_dataset, criterion, optimizer, num_epochs=100):
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)

    best_acc = 0.0
    print("start training")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'model.pth')

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        #if epoch % 2 == 1:
        scheduler.step()

# train_model(model, train_data, val_data, criterion, optimizer, num_epochs=120)


class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.images[idx]

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

test_dataset = TestDataset('data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

checkpoint = torch.load('model_dist.pth')

# Adjust the keys
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in checkpoint['model_state_dict'].items():
    name = k[7:] if k.startswith('module.') else k  # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.to(device)
"""https://github.com/Burf/ModelSoups/blob/main/usage_torch.ipynb"""
import model_soup
import numpy as np
val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
def metric(y_true, y_pred):
    _, preds = torch.max(y_pred, 1)
    corrects = torch.sum(preds == y_true).item()
    accuracy = corrects / len(y_true)
    return accuracy

def evaluate(model, data_loader, metric, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    metric_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch = metric(labels, outputs)
            metric_sum += batch * inputs.size(0)
            total_samples += inputs.size(0)

    average = metric_sum / total_samples
    return average

best_score = 0
print("\n[Greedy Soup (uniform weight update) Performance]") #original paper style
greedy_model = model_soup.torch.greedy_soup(model, "model_dist.pth", val_loader, metric = metric, device = device, compare = np.greater_equal, update_greedy = False)
score = evaluate(greedy_model, val_loader,metric, device=device)
print("score : {0:.4f}".format(score))
if score > best_score: model = greedy_model 
 
print("\n[Greedy Soup (greedy weight update) Performance]")
greedy_model = model_soup.torch.greedy_soup(model, "model_dist.pth", val_loader, metric = metric, device = device, compare = np.greater_equal, update_greedy = True)
score = evaluate(greedy_model, val_loader,metric, device=device)
print("score : {0:.4f}".format(score))
if score > best_score: model = greedy_model

print("\n[Uniform Soup Performance]")
uniform_model = model_soup.torch.uniform_soup(model, "model_dist.pth", device = device)
score = evaluate(uniform_model, val_loader,metric, device=device)
print("score : {0:.4f}".format(score))
if score > best_score: model = uniform_model

model.eval() 
predictions = []

with torch.no_grad():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(zip(filenames, preds.cpu().numpy()))

final_predictions = [(file, f'a1_{pred}') for file, pred in predictions]
submission_df = pd.DataFrame(final_predictions, columns=['id', 'label'])
submission_df.to_csv('submission.csv', index=False)

