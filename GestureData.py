import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset

class Dataset(Dataset):
    def __init__(self, root="./data", train=True, transforms=None):
        self.root = root
        self.pre = "/train_set/" if train else "/test_set/"
        self.count = 0
        self.labels = []
        self.data = []
        self.nums = [4468, 4381, 4254] if train else [865, 899, 878]
        self.names = ["O/","V/","W/"]
        for i in range(3):
            name = self.names[i]
            for j in range(self.nums[i]):
                self.data.append(self.read_image(self.root+self.pre+name+str(j)+".jpg"))
                self.labels.append(i)
                self.count += 1

    def read_image(self, file_name):
        image = Image.open(file_name)
        image = torchvision.transforms.functional.pil_to_tensor(image)
        return image

    def __getitem__(self, index):
        image = self.data[index]
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[index]
        return (image, label)

    def __len__(self):
        return self.count