import torch
import torchvision.models as models
import torch.quantization

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(1024,3)
state_dict = torch.load("Trained_Models_test\gesture_mn3s_V1.pth")

model.load_state_dict(state_dict)

model.eval()

qconfig = torch.quantization.get_default_qconfig('fbgemm')  

model.qconfig = qconfig
torch.quantization.prepare(model, inplace=True)

model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

torch.save(model, "Trained_Models_test\gesture_mn3s_V1_quantized.pth")