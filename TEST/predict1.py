import cv2
import time
import torch
#import torchvision.transforms as transforms
from torchvision.transforms import v2
import numpy as np
import os
#import pandas
import torchvision
import torch.nn as nn
#import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
#from torch.utils.data.sampler import SubsetRandomSampler
#from torch.utils.data.dataset import Dataset
from torchvision.models import resnet18
#from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, ResNet50_Weights

# 加载模型
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
"""
model = resnet18(pretrained=False)

model.fc = nn.Linear(model.fc.in_features, 3)
Model_path = './gesture_resnet18_V2.pth'
model.load_state_dict(torch.load(Model_path, map_location = torch.device('cpu')))
"""
#model.fc = torch.nn.Linear(512, 18)
#checkpoint = torch.load("./ResNet18.pth")
#model.load_state_dict(checkpoint['MODEL_STATE'])

model = torchvision.models.mobilenet_v3_small()
model.classifier[3] = torch.nn.Linear(1024, 3)
model = torch.load("Trained_Models_test\gesture_mn3s_V1_quantized.pth",map_location = torch.device('cpu'))

#model = torch.jit.script(model)
model.eval()

# 创建 VideoCapture 对象
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

# 定义图像转换
transform = v2.Compose([
    v2.PILToTensor(),
    #v2.Resize((112, 112)),
    v2.ToDtype(torch.float32, scale=True),
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

classes = ['rock', 'scissors', 'paper']

top_left = (30,30)
bottom_right = (142,142)

last = time.time()
count = 0
while True:
    count += 1
    if count % 10 == 0:
        now = time.time()
        print(10 / (now - last))
        last = now
    
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # 获取矩形区域的图像
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = Image.fromarray(roi)

    # 将图像转换为模型需要的格式
    image = transform(roi).unsqueeze(0)

    # 运行模型
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    """
    # start = time.time()
    output = model(image)
    _, predicted = torch.max(output, 1)
    # end = time.time()
    # print("Time: {:.6f}s".format(end - start))
    """
    # 显示预测结果
    cv2.putText(frame, 'Predicted: {}'.format(classes[predicted.item()]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    # 如果按下 'q' 键，就退出循环
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 释放 VideoCapture 对象并关闭窗口
cap.release()
cv2.destroyAllWindows()