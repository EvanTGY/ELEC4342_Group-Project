import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
import mediapipe as mp
import numpy as np
import time

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.model(x)
    
def get_hand_bbox(hand_landmarks, image_width, image_height, padding=20):
    x = [landmark.x for landmark in hand_landmarks.landmark]
    y = [landmark.y for landmark in hand_landmarks.landmark]
    x_min = max(0, int(min(x) * image_width) - padding)
    x_max = min(image_width - 1, int(max(x) * image_width) + padding)
    y_min = max(0, int(min(y) * image_height) - padding)
    y_max = min(image_height - 1, int(max(y) * image_height) + padding)
    return x_min, y_min, x_max, y_max


def main():
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    model = ResNet18()
    model.load_state_dict(torch.load('quantized_resnet18_weights.pth', map_location=torch.device('cpu')))
    model.eval()

    fps_timer = time.time()
    fps_counter = 0

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    frame_interval = 3
    frame_count = 0

    while True:
        ret, frame = cap.read()
        fps_counter += 1
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter / (time.time() - fps_timer)
            fps_timer = time.time()
            fps_counter = 0

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.GaussianBlur(img_rgb, (5, 5), 0) 

        results = mp_hands.process(img_rgb)

        blank_img = np.zeros_like(frame)

        if frame_count % frame_interval == 0:
            hand_landmarks = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(blank_img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3))

            if hand_landmarks is not None:
                x_min, y_min, x_max, y_max = get_hand_bbox(hand_landmarks, frame.shape[1], frame.shape[0], padding=20)
                hand_image = blank_img[y_min:y_max, x_min:x_max]

                pil_image = Image.fromarray(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB))
                input_image = transform(pil_image)
                input_image = input_image.unsqueeze(0)

                with torch.no_grad():
                    output = model(input_image)
                _, predicted_class = torch.max(output, 1)
                class_names = ['rock', 'scissor', 'paper']
                class_label = class_names[predicted_class.item()]

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Class: {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Classification', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()