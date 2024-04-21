from matplotlib import transforms
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch.nn.functional as F
from torchinfo import summary
import os
import glob
import random
import csv


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3,96,kernel_size=11,stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(96,256,kernel_size=5,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.fc1 = nn.Linear(6*6*256,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,1000)
        self.fc4 = nn.Linear(1000,3)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(-1,6*6*256)  #拉平
        x = F.relu(F.dropout(self.fc1(x),0.5))
        x = F.relu(F.dropout(self.fc2(x),0.5))
        x = F.dropout(self.fc3(x),0.5)
        output = self.fc4(x)
        return output
    

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
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # prevent all data are separated btw 0~1
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


def run(epoch, model, Device):
    test_losses = []
    test_accuracies = []
    best_accuracy = 0
    for i in range(epoch):
        train(model, train_loader, Device)
        avg_test_loss, test_accuracy = test(model, test_loader, Device)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     torch.save(model.state_dict(), 'best.pth')
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        print(f"Epoch {i+1}: Loss = {avg_test_loss:.4f}, Accuracy = {test_accuracy:.2f}%")
    print("The best accuracy: ", best_accuracy)
    torch.save(model.state_dict(), 'final.pth')
    return test_losses, test_accuracies


if __name__ == '__main__':
    # args
    learning_rate = 0.015
    criterion = nn.CrossEntropyLoss()
    epoch = 25
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset
    train_db = dataset('dataset', 227, mode='train')
    test_db = dataset('dataset', 227, mode='test')
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_db, batch_size=batch_size, num_workers=2)
    # model
    model = AlexNet()
    print("Learning rate: ", learning_rate)
    before_test_losses, before_test_accuracies = run(epoch, model, device)
