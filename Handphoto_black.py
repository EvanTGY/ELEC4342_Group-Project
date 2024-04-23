import cv2
import os
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4)
mp_draw = mp.solutions.drawing_utils

input_folder_path = "./data_marked/test_set/Scissor"
output_folder_path = "./data_black/test_set/Scissor"

os.makedirs(output_folder_path, exist_ok=True)

image_count = 0
for filename in os.listdir(input_folder_path):
    if filename.endswith('.jpg'):
        # read image
        img_path = os.path.join(input_folder_path, filename)
        img = cv2.imread(img_path)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # keypoint detection
        results = mp_hands.process(img_rgb)

        # create a blank image to draw keypoints and connections
        blank_img = np.zeros_like(img)

        # plot the point and connection line on the blank image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(blank_img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2))

            # save image
            output_filename = f'{image_count}.jpg'
            output_img_path = os.path.join(output_folder_path, output_filename)
            cv2.imwrite(output_img_path, blank_img)

            print(f'Saved output image: {output_img_path}')

            image_count += 1

mp_hands.close()