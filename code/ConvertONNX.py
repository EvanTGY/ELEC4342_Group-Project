import torch
import torchvision
import torch.nn as nn

# 加载模型
model = torchvision.models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model.load_state_dict(torch.load(r"2023-2024 HKU CE SOURCE(Yr2)\ELEC4342\ELEC4342_Group-Project\Trained_Models_final\ResNet18_Marked_96.pth"))
model.eval()

# 创建一个虚拟输入
input_tensor = torch.randn(128, 3, 224, 224)

# 导出模型
torch.onnx.export(model, input_tensor, "ResNet18_ACC_96.onnx")