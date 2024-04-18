import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet5(nn.Module):
    pass





