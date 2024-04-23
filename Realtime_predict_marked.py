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
from torchvision.models import resnet50, resnet18, resnet34
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, ResNet50_Weights, ResNet18_Weights
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import mediapipe as mp
# 加载模型
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
# model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
Model_path = 'Trained_Models_final/ResNet34_Marked.pth'
model.load_state_dict(torch.load(Model_path))
model.eval()

# 初始化 MediaPipe 手部解决方案
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 初始化 MediaPipe 绘图功能
mp_draw = mp.solutions.drawing_utils

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

top_left = (150,100)
bottom_right = (450,400)

pre_frame_time = 0

while True:
    ret,frame = cap.read()
    if not ret:
        break
    current_time = time.time()
    fps = 1/(current_time - pre_frame_time)
    pre_frame_time = current_time
    cv2.putText(frame, 'FPS: {}'.format(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)  # 可选：使用高斯模糊进行平滑处理
    
    results = hands.process(rgb_image)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3))
        hand_landmarks = [[int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])] for landmark in hand_landmarks.landmark]
        hand_landmarks = np.array(hand_landmarks)

        # top_left = np.min(hand_landmarks, axis=0)
        # bottom_right = np.max(hand_landmarks, axis=0)

        # 计算矩形框的坐标，并向外扩展20像素
        padding = 20
        top_left = np.maximum(np.min(hand_landmarks, axis=0) - padding, 0)
        bottom_right = np.minimum(np.max(hand_landmarks, axis=0) + padding, [frame.shape[1], frame.shape[0]])

        if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] <= frame.shape[1] and bottom_right[1] <= frame.shape[0]:
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # 将 BGR 图像转换为 RGB 图像
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            rgb_roi= cv2.GaussianBlur(rgb_roi, (5, 5), 0)
            
            # 将图像转换为模型需要的格式
            image = transform(rgb_roi).unsqueeze(0)

                        # 运行模型
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)

            # 在原始图像上绘制矩形框
            cv2.rectangle(frame, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0, 255, 0), 2)
        
            # 显示预测结果
            cv2.putText(frame, 'Predicted: {}'.format(classes[predicted.item()]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        else:
                print("ROI coordinates are out of bounds.")
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# while True:

#     current_time = time.time()
#     # 读取一帧
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # 获取矩形区域的图像
#     roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

#     # 将 BGR 图像转换为 RGB 图像
#     rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     rgb_image= cv2.GaussianBlur(rgb_image, (5, 5), 0)  # 可选：使用高斯模糊进行平滑处理
    
#     # 执行手部关键点检测
#     results = hands.process(rgb_image)
    
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
#                                 mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
#                                 mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3))

#     cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

#     #     将图像转换为模型需要的格式
#     image = transform(frame).unsqueeze(0)

#     # 运行模型
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)

#     # 显示预测结果
#     cv2.putText(frame, 'Predicted: {}'.format(classes[predicted.item()]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('frame', frame)

#     # 如果按下 'q' 键，就退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放 VideoCapture 对象并关闭窗口
# cap.release()
# cv2.destroyAllWindows()