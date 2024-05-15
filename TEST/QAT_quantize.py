from matplotlib import transforms
import torchvision
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.models import resnet18
import torchvision.models as models
import torch.nn.functional as F
import os
import glob
import random
import csv
import torch.onnx
from torchvision.models.resnet import resnet18, ResNet18_Weights



class dataset(Dataset):
    def __init__(self, root, resize, mode):
        super(dataset, self).__init__()
        self.root = root
        self.resize = resize

        self.nametolabel = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.nametolabel[name] = len(self.nametolabel.keys())
        print(self.nametolabel)
        self.images, self.labels = self.load_csv('images.csv')
        print("csv file loaded")
        # cut the training, testing set
        if mode=='train': #80%
            self.images = self.images[:int(0.8*len(self.images))]
            self.labels = self.labels[:int(0.8*len(self.labels))]
        else: #20% = 80%->100%
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

        # image, label
    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.nametolabel.keys():
                # 'dataset\\O\\0.jpg'
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'dataset\\O\\1.jpg'
                    name = img.split(os.sep)[-2]
                    label = self.nametolabel[name]
                    writer.writerow([img, label])
        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] -> [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'dataset\\O\\0.jpg'
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        transform = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path -> image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomVerticalFlip(), 
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # prevent all data are separated btw 0~1
        ])
        img = transform(img)
        label = torch.tensor(label)
        return img, label


def train(model, train_loader, Device):
    total_train_step = 0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.train()
    for batch_index, (data , target) in enumerate(train_loader):
        model = model.to(Device)
        data = data.to(Device)
        target = target.to(Device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1


def test(model, test_loader, Device):
    correct = 0
    total = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(Device), target.to(Device)
            output = model(data)
            loss = criterion(output, target)
            test_loss = test_loss + loss.item()
            predicted = output.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
    total_accuracy = 100 * correct / total
    return test_loss, total_accuracy


def run(epoch, model, Device, learning_rate):
    test_losses = []
    test_accuracies = []
    best_accuracy = 0
    for i in range(epoch):
        train(model, train_loader, Device)
        avg_test_loss, test_accuracy = test(model, test_loader, Device)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'resnet18_dataset5_retrain_qat.pth')
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        print(f"Epoch {i+1}: Loss = {avg_test_loss:.4f}, Accuracy = {test_accuracy:.2f}%")
    print("The best accuracy: ", best_accuracy)
    # torch.save(model.state_dict(), 'final.pth')
    return test_losses, test_accuracies


if __name__ == '__main__':
    # args
    learning_rate = 0.012
    criterion = nn.CrossEntropyLoss()
    epoch = 20
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset
    train_db = dataset('dataset5', 227, mode='train')
    test_db = dataset('dataset5', 227, mode='test')
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_db, batch_size=batch_size, num_workers=4)
    # model
    # model = ResNet18()
    # model = models.mobilenet_v2(pretrained=True)
    # num_ftrs = model.classifier[1].in_features
    # model.classifier = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(num_ftrs, 3)
    # )
    # for param in model.parameters():
    #     param.requires_grad = False
    # in_channels = model.classifier[1].in_channels
    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Conv2d(in_channels, 3, kernel_size=1),
    #     nn.ReLU(inplace=True),
    #     nn.AdaptiveAvgPool2d((1, 1))
    # )
    # print(model)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 3)

    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 3)
    # mobile_model = MobileNetV3Model(num_classes=3)
    
    model = torchvision.models.quantization.resnet18(weights="DEFAULT", quantize=False)
    model.fc = torch.nn.Linear(512, 3)
    model.load_state_dict(torch.load("resnet18_dataset5_retrain.pth"))
    model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
    model.fuse_model(is_qat=True)
    model_qat = torch.quantization.prepare_qat(model, inplace=False)

    # model = torch.quantization.convert(model_qat.eval(), inplace=False)
    print("Learning rate: ", learning_rate)
    before_test_losses, before_test_accuracies = run(epoch, model_qat, device, learning_rate)