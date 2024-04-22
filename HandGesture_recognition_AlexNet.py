import pandas
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import models
import torch.nn.functional as F

batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.model = nn.Sequential( #3 32 32
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),# 6 32 32
            nn.MaxPool2d(kernel_size=2, stride=2, padding =0),#6 16 16
            nn.BatchNorm2d(num_features=6),
            nn.CELU(inplace=True),
            
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1, padding=2),# 12 16 16
            nn.MaxPool2d(kernel_size=2, stride=2, padding =0), #12 8 8
            nn.BatchNorm2d(num_features=12),
            nn.CELU(inplace=True),
            
            nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, stride=1, padding=1),# 8 8 8
            nn.BatchNorm2d(num_features=8),
            nn.CELU(inplace=True),

            nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, stride=1, padding=1),# 6 8 8
            nn.BatchNorm2d(num_features=6),
            nn.CELU(inplace=True),

            nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3, stride=1, padding=1),# 4 8 8
            nn.MaxPool2d(kernel_size=2, stride=2, padding =0), # 4 4 4
            nn.BatchNorm2d(num_features=4),
            nn.CELU(inplace=True),
            )

        self.liner = nn.Sequential(

            nn.Linear(3136,4*4*4),
            nn.CELU(inplace=True),

            nn.Linear(4*4*4,16),
            nn.Tanh(),

            nn.Linear(16,4)
        )
        
    def forward(self, img):
        
        x = self.model(img)

        z = self.liner(x.view(x.shape[0], -1))

        return z

class SaveImagesToCSV:
    def __init__(self,root="./data", train = True, transforms=None):
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
                self.data.append(self.root+self.pre+name+str(j)+".jpg")
                self.labels.append(i)
                self.count += 1
        
        df = pandas.DataFrame({'image_path': self.data, 'label': self.labels})
        if train:
            df.to_csv('./data/train_set/train_images.csv', index=False)
        else:
            df.to_csv('./data/test_set/test_images.csv', index=False)
    
class Dataset(Dataset):
    def __init__(self, csv_file, transforms = None):
        if not os.path.exists(csv_file):
            print('CSV file not found')
            return
        self.dataframe = pandas.read_csv(csv_file)
        self.transform = transforms

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.dataframe.iloc[idx, 1]
        return image, label

    

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


if __name__ == '__main__':
    
    transformations= transforms.Compose([
                    transforms.Resize((227,227)),
                    transforms.RandomRotation(180),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])
    save_train_images = SaveImagesToCSV(root="./data", train=True)
    save_test_images = SaveImagesToCSV(root="./data", train=False)

    train_dataset = Dataset(csv_file='./data/train_set/train_images.csv', transforms=transformations)
    test_dataset = Dataset(csv_file="./data/test_set/test_images.csv", transforms=transformations)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    trained_model_path = 'Trained_models/model_AlexNet.pth'

    if os.path.exists(trained_model_path):
        print ('Loading model from {}'.format(trained_model_path))
        model = AlexNet()
        model.load_state_dict(torch.load(trained_model_path))
        print ('Model loaded')
        model = model.to(device)
    else:
        print ('Training model')
        model = AlexNet()
        model = model.to(device)

    best_accuracy = 0.0

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(1, 21):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)  
        schedular.step()
        print ('learning rate: {}'.format(optimizer.param_groups[0]['lr']))

    torch.save(model.state_dict(), 'Trained_Models/model_AlexNet.pth')
    print('Model saved')
