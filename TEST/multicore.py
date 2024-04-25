import cv2
import torch
import threading
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import mediapipe as mp
import numpy as np
import time


class CameraThread(threading.Thread):
    def __init__(self, cap, frame_buffer):
        super().__init__()
        self.cap = cap
        self.frame_buffer = frame_buffer

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_buffer.append(frame)



class ModelThread(threading.Thread):
    def __init__(self, frame_buffer, model, transform):
        super().__init__()
        self.frame_buffer = frame_buffer
        self.model = model
        self.transform = transform
        self.hands = mp.solutions.hands.Hands
    def run(self):
        while True:
            if len(self.frame_buffer) > 0:
                frame = self.frame_buffer.pop(0)
                
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_image = cv2.GaussianBlur(rgb_image, (5, 5), 0)

    
                results = self.hands.process(rgb_image)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_landmarks = [[int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])] for landmark in hand_landmarks.landmark]
                    hand_landmarks = np.array(hand_landmarks)

                    padding = 20
                    top_left = np.maximum(np.min(hand_landmarks, axis=0) - padding, 0)
                    bottom_right = np.minimum(np.max(hand_landmarks, axis=0) + padding, [frame.shape[1], frame.shape[0]])

                    if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] <= frame.shape[1] and bottom_right[1] <= frame.shape[0]:
                        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


                        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        rgb_roi = cv2.GaussianBlur(rgb_roi, (5, 5), 0)

                
                        image = self.transform(rgb_roi).unsqueeze(0)

                    
                        with torch.no_grad():
                            output = self.model(image)
                            _, predicted = torch.max(output, 1)

       
                        print('Predicted:', predicted.item())
def main():
    