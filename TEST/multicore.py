import cv2
import os
import torch
import threading
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock, ResNet50_Weights, ResNet18_Weights
import mediapipe as mp
import numpy as np
import time
import threading

# 加载模型
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# model = resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
Model_path = 'ResNet18_Marked_96.pth'
# Model_path = 'Trained_Models_final/ResNet50_Marked.pth'
model.load_state_dict(torch.load(Model_path, map_location=torch.device('cpu')))
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



class CameraThread(threading.Thread):
    def __init__(self, cap, frame_buffer, lock):
        super().__init__()
        self.cap = cap
        self.frame_buffer = frame_buffer
        self.lock = lock
        self.hands = mp.solutions.hands.Hands()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)
            results = self.hands.process(rgb_image)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,   
                                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3))
                with self.lock:
                    self.frame_buffer.append(frame)

    def stop(self):
        self.running = False

class ModelThread(threading.Thread):
    def __init__(self, frame_buffer, model, transform, lock):
        super().__init__()
        self.frame_buffer = frame_buffer
        self.model = model
        self.transform = transform
        self.lock = lock
        self.running = True
        self.predicted = torch.tensor(0)

    def run(self):
        while self.running:
            with self.lock:
                if len(self.frame_buffer) > 0:
                    frame = self.frame_buffer.pop(0)
                    image = transform(frame).unsqueeze(0)
                    with torch.no_grad():
                        output = self.model(image)
                        _, self.predicted = torch.max(output, 1)
                    cv2.putText(frame, 'Predicted: {}'.format(classes[self.predicted.item()]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    def stop(self):
        self.running = False

def main():
    cap = cv2.VideoCapture(0)
    frame_buffer = []
    lock = threading.Lock()

    model_thread = ModelThread(frame_buffer, model, transform, lock)
    model_thread.start()

    camera_thread = CameraThread(cap, frame_buffer, lock)
    camera_thread.start()

    camera_thread.join()
    model_thread.stop()
    model_thread.join()

    cap.release()

if __name__ == '__main__':
    main()