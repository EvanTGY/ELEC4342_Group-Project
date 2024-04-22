import cv2
import os
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# 改变工作目录到脚本所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

input_folder_path = "./data_set/train_set/V"
output_folder_path = "./data_marked/train_set/Scissor"

os.makedirs(output_folder_path, exist_ok=True)

image_count = 0
for filename in os.listdir(input_folder_path):
    if filename.endswith('.jpg'):
        # read image
        img_path = os.path.join(input_folder_path, filename)
        img = cv2.imread(img_path)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.GaussianBlur(img_rgb, (5, 5), 0)  # 可选：使用高斯模糊进行平滑处理

        # keypoint detection
        results = mp_hands.process(img_rgb)

        # plot the point and connection line
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2))

            # save image
            output_filename = f'{image_count}.jpg'
            output_img_path = os.path.join(output_folder_path, output_filename)
            cv2.imwrite(output_img_path, img)

            print(f'Saved output image: {output_img_path}')

            image_count += 1


mp_hands.close()