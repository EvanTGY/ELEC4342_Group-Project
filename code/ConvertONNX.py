import torch
import torchvision

# 加载模型
model = torchvision.models.resnet18()
model.load_state_dict(torch.load("resnet18-5c106cde.pth")
model.eval()

# 创建一个虚拟输入
input_tensor = torch.randn(128, 3, 224, 224)

# 导出模型
torch.onnx.export(model, input_tensor, "model.onnx")