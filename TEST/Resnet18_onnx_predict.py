import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import onnxruntime as ort
import mediapipe as mp
import numpy as np
import time,os
from torchvision.models.resnet import resnet18, ResNet18_Weights


    
def get_hand_bbox(hand_landmarks, image_width, image_height, padding=20):
    x = [landmark.x for landmark in hand_landmarks.landmark]
    y = [landmark.y for landmark in hand_landmarks.landmark]
    x_min = max(0, int(min(x) * image_width) - padding)
    x_max = min(image_width - 1, int(max(x) * image_width) + padding)
    y_min = max(0, int(min(y) * image_height) - padding)
    y_max = min(image_height - 1, int(max(y) * image_height) + padding)
    return x_min, y_min, x_max, y_max


def main():

    sess = ort.InferenceSession(r"E:\OneDrive - The University of Hong Kong - Connect\2023-2024 HKU CE SOURCE(Yr2)\ELEC4342\ELEC4342_Group-Project\Trained_Models_test\ResNet18_128.onnx")

    mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils


 
    fps_timer = time.time()
    fps_counter = 0

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    frame_interval = 4
    frame_count = 0
    class_label = None
    x_min_display = 0 
    x_max_display = 0 
    y_min_display = 0 
    y_max_display = 0

    while True:
        ret, frame = cap.read()
        fps_counter += 1
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_counter / (time.time() - fps_timer)
            fps_timer = time.time()
            fps_counter = 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img_rgb = cv2.GaussianBlur(img_rgb, (5, 5), 0) 

        results = mp_hands.process(img_rgb)

        blank_img = np.zeros_like(frame)

        if frame_count % frame_interval == 0:
            x_min_display, y_min_display, x_max_display, y_max_display = 0, 0, 0, 0
            class_label = None

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
                    input_name = sess.get_inputs()[0].name
                    output = sess.run(None, {input_name: input_image.numpy()})
                output_tensor = torch.tensor(output[0])
                _, predicted_class = torch.max(output_tensor, 1)
                class_names = ['rock', 'scissor', 'paper']
                class_label = class_names[predicted_class.item()]
                x_min_display = x_min
                x_max_display = x_max
                y_min_display = y_min 
                y_max_display = y_max
        
        cv2.rectangle(frame, (x_min_display, y_min_display), (x_max_display, y_max_display), (0, 255, 0), 1)
        class_label_display = class_label
        cv2.putText(frame, f"Class: {class_label_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Classification', frame)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break
        cv2.waitKey(1)
    # cap.release()
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    main()