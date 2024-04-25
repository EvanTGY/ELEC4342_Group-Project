import numpy as np
import os
import pandas
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision.models import resnet50, resnet18, resnet34
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, ResNet50_Weights, ResNet18_Weights
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import mediapipe as mp
import cv2
import torch
import torchvision.transforms as transforms
from picamera import PiCamera
from picamera.array import PiRGBArray

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 3)
Model_path = 'E:\TEST\ResNet18_Marked_96.pth'
# 加载模型
model.load_state_dict(torch.load(Model_path), map_location=torch.device('cpu'))
model.eval()


# 初始化 MediaPipe 手部解决方案
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 初始化 MediaPipe 绘图功能
mp_draw = mp.solutions.drawing_utils

# 创建 PiCamera 对象
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# 定义图像转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 循环捕获图像并进行预测
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)

    # 运行模型
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # 显示预测结果
    print('Predicted:', predicted.item())

    # 清除流以准备下一帧
    rawCapture.truncate(0)

    #如果你想在按下某个键后停止，可以取消注释以下代码
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
camera.close()