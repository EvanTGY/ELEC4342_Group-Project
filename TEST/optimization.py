from torchvision.models import resnet18
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=True) 
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3) 

    def forward(self, x):
        return self.model(x)


model = ResNet18()
model.load_state_dict(torch.load('ResNet18_optimized.pth'))
model.eval()


conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
fc_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]

for layer in conv_layers:
    prune.l1_unstructured(layer, name="weight", amount=0.2)

for layer in fc_layers:
    prune.l1_unstructured(layer, name="weight", amount=0.2)

for layer in conv_layers:
    prune.remove(layer, "weight")

for layer in fc_layers:
    prune.remove(layer, "weight")