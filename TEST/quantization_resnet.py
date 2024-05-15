import torch
import torchvision.models as models
import torch.quantization
from torchvision.models.resnet import resnet18, ResNet18_Weights
import torch.nn as nn


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

# 加载状态字典
state_dict = torch.load(r'Trained_Models_test\resnet18_dataset5_retrain.pth')
model.load_state_dict(state_dict)

model.eval()


model.qconfig = torch.quantization.get_default_qconfig('fbgemm')


model = torch.quantization.prepare(model, inplace=True)


input_data = torch.rand(1, 3, 227, 227)
model(input_data)


model = torch.quantization.convert(model, inplace=True)

torch.save(model.state_dict(), 'Trained_Models_test/quantized1_resnet18.pth')