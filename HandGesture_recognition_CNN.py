import torch
import os
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset



batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(56320, 4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Dataset(Dataset):
    def __init__(self, root="./data", train=True, transforms=None):
        self.root = root
        self.pre = "/train_set/" if train else "/test_set/"
        self.count = 0
        self.labels = []
        self.data = []
        self.nums = [4468, 4381, 4254] if train else [865, 899, 878]
        self.names = ["O/","V/","W/"]
        self.transforms = transforms
        for i in range(3):
            name = self.names[i]
            for j in range(self.nums[i]):
                self.data.append(self.read_image(self.root+self.pre+name+str(j)+".jpg"))
                self.labels.append(i)
                self.count += 1

    def read_image(self, file_name):
        with Image.open(file_name) as image:
        # image = torchvision.transforms.functional.pil_to_tensor(image)
            return image.copy()

    def __getitem__(self, index):
        image = self.data[index]
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return (image, label)

    def __len__(self):
        return self.count

criterion = nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer,epoch):
    model.train()
    train_loss = 0
    for (data, target) in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        torch.cuda.empty_cache()

    train_loss /= len(train_loader.dataset)
    print ('Epoch: {}'.format(epoch))
    print('Average train Loss: {:.6f}'.format(train_loss))
    return train_loss


def test(model, device, test_loader):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            predict = output.argmax(dim=1, keepdim=True)
            correct += predict.eq(target.view_as(predict)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Average test Loss: {:.6f}'.format(test_loss))
    print('Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy


if __name__ == '__main__':
    
    transformations= transforms.Compose([
                    transforms.RandomRotation(180),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])

    train_dataset = Dataset(root="./data", train=True, transforms=transformations)
    test_dataset = Dataset(root="./data", train=False, transforms=transformations)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print('Data loaded')

    trained_model_path = 'Trained_Models/model_Vgg16_CNN.pth'

    if os.path.exists(trained_model_path):
        model = ConvNet()
        model.load_state_dict(torch.load(trained_model_path))
        model = model.to(device)
    else:
        model = ConvNet()

        model = model.to(device)

    best_accuracy = 0.0

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 10):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'Trained_Models/model_Vgg16_CNN.pth')
        print('Model saved')
