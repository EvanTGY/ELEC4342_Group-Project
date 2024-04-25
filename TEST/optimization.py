from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn




model = resnet18(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  
model.load_state_dict(torch.load('Trained_Models_test/ResNet18_Marked_96.pth'))
model.eval()


conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
fc_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]

for layer in conv_layers:
    prune.l1_unstructured(layer, name="weight", amount=0.3)

for layer in fc_layers:
    prune.l1_unstructured(layer, name="weight", amount=0.3)

for layer in conv_layers:
    prune.remove(layer, "weight")

for layer in fc_layers:
    prune.remove(layer, "weight")

print("OK")
