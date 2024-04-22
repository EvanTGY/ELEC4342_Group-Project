import cv2
import time
import torch
import torchvision.transforms as transforms
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
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, ResNet50_Weights

# 加载模型
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
Model_path = 'Trained_Models_final/model_ResNet50_best.pth'
model.load_state_dict(torch.load(Model_path))
model.eval()

# 创建 VideoCapture 对象
cap = cv2.VideoCapture(0)

# 定义图像转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

classes = ['rock', 'scissors', 'paper']

top_left = (100,100)
bottom_right = (400,400)

pre_frame_time = 0
while True:
    current_time = time.time()
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # 获取矩形区域的图像
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # 将图像转换为模型需要的格式
    image = transform(roi).unsqueeze(0)

    # 运行模型
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # 显示预测结果
    cv2.putText(frame, 'Predicted: {}'.format(classes[predicted.item()]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    # 如果按下 'q' 键，就退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放 VideoCapture 对象并关闭窗口
cap.release()
cv2.destroyAllWindows()