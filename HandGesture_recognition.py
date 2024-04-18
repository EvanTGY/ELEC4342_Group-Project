import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet5(nn.Module):
    pass





def dataset():
    train_data_folder = './data/train_set'
    test_data_folder = './data/test_set'

    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomRotation(15),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)
                )])
    train_data = datasets.ImageFolder (root = train_data_folder, transform = transform)
    test_data = datasets.ImageFolder (root = test_data_folder, transform = transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = True)
    
    print ('Data loaded successfully')
    return train_loader, test_loader