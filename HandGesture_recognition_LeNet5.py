import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import models
import pandas
import os


batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class LeNet5_unquantized(nn.Module):
    def __init__(self):
        super(LeNet5_unquantized, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(44944, 120)
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


class SaveImagesToCSV:
    def __init__(self,root="./data_marked", train = True, transforms=None):
        self.root = root
        self.pre = "/train_set/" if train else "/test_set/"
        self.count = 0
        self.labels = []
        self.data = []
        #self.nums = [4468, 4381, 4254] if train else [865, 899, 878]
        self.nums = [3960, 3951, 3792] if train else [721, 874 ,812]
        self.names = ["Rock/","Scissor/","Paper/"]
        #self.names =['O/','v','W/]
        self.transforms = transforms
        for i in range(3):
            name = self.names[i]
            for j in range(self.nums[i]):
                self.data.append(self.root+self.pre+name+str(j)+".jpg")
                self.labels.append(i)
                self.count += 1
        
        df = pandas.DataFrame({'image_path': self.data, 'label': self.labels})
        if train:
            df.to_csv('./data_marked/train_set/train_images_marked.csv', index=False)
        else:
            df.to_csv('./data_marked/test_set/test_images_marked.csv', index=False)
    
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
    print()
    return test_loss, accuracy



if __name__ == '__main__':
    
    transformations= transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])
    save_train_images = SaveImagesToCSV(root="./data_marked", train=True)
    save_test_images = SaveImagesToCSV(root="./data_marked", train=False)

    train_dataset = Dataset(csv_file='./data_marked/train_set/train_images_marked.csv', transforms=transformations)
    test_dataset = Dataset(csv_file="./data_marked/test_set/test_images_marked.csv", transforms=transformations)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Data loaded')

    model = LeNet5_unquantized()
    # model = model.to(device)    

    trained_model_path = 'Trained_Models_test/LetNet5_marked.pth'
    if os.path.exists(trained_model_path):
        print('Loading model from {}'.format(trained_model_path))
        model.load_state_dict(torch.load(trained_model_path))
        print('Model loaded')
        model = model.to(device)
    else:
        print('Training model')
        model = model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    lowerst_test_loss = float('inf')

    for epoch in range(1, 11):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        scheduler.step()
        if test_loss < lowerst_test_loss:
            lowerst_test_loss = test_loss
            torch.save(model.state_dict(), 'Trained_Models_test/LeNet5_Marked.pth')
            print('Model saved')
            print('Model saved with test loss: {:.6f}'.format(test_loss))
            print('Model saved with accuracy: {:.2f}%'.format(accuracy))
            print()

        torch.cuda.empty_cache()