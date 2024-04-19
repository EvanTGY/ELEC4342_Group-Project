import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import models


batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LeNet5_unquantized(nn.Module):
    def __init__(self):
        super(LeNet5_unquantized, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(218544, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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


# if __name__ == '__main__':
    
#     transformations= transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#     train_dataset = Dataset(root="./data", train=True, transforms=transformations)
#     test_dataset = Dataset(root="./data", train=False, transforms=transformations)

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
#     print('Data loaded')

#     # model = LeNet5_unquantized().to(device)

#     model = models.vgg16(pretrained=True)


#     num_ftrs = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(num_ftrs, 3)

#     model = model.to(device)
    
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(1, 10):
#         train_loss = train(model, device, train_loader, optimizer, epoch)
#         test_loss, accuracy = test(model, device, test_loader)

#     torch.save(model.state_dict(), 'model.pth')
#     print('Model saved')


if __name__ == '__main__':
    
    transformations= transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = Dataset(root="./data", train=True, transforms=transformations)
    test_dataset = Dataset(root="./data", train=False, transforms=transformations)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print('Data loaded')

    model = LeNet5_unquantized().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 10):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)

    torch.save(model.state_dict(), 'Trained_Models/model_LeNet5.pth')
    print('Model saved')