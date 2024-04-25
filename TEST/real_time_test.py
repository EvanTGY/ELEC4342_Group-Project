import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
import time
import torchvision.models as models

# class AlexNet(nn.Module):
#     def __init__(self):
#         super(AlexNet, self).__init__()
#         self.conv1 = nn.Conv2d(3,96,kernel_size=11,stride=4)
#         self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
#         self.conv2 = nn.Conv2d(96,256,kernel_size=5,padding=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
#         self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1)
#         self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1)
#         self.conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
#         self.fc1 = nn.Linear(6*6*256,4096)
#         self.fc2 = nn.Linear(4096,4096)
#         self.fc3 = nn.Linear(4096,1000)
#         self.fc4 = nn.Linear(1000,3)
    
#     def forward(self,x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = self.pool3(x)
#         x = x.view(-1,6*6*256)  #拉平
#         x = F.relu(F.dropout(self.fc1(x),0.5))
#         x = F.relu(F.dropout(self.fc2(x),0.5))
#         x = F.dropout(self.fc3(x),0.5)
#         output = F.softmax(self.fc4(x),dim=1)
#         return output


# model = AlexNet()
# model.load_state_dict(torch.load('highestAcc2.pth'))
# model.eval()

model = models.squeezenet1_0(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
in_channels = model.classifier[1].in_channels
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(in_channels, 3, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
)
model.load_state_dict(torch.load('squeezenet1_0_dataset_enhanced2.pth'))
model.eval()

fps_timer = time.time()
fps_counter = 0

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        fps = fps_counter / (time.time() - fps_timer)
        print(fps)
        fps_timer = time.time()
        fps_counter = 0

    frame_height, frame_width, _ = frame.shape
    box_size = 400
    start_x = int((frame_width - box_size) / 2)
    start_y = int((frame_height - box_size) / 2)
    end_x = start_x + box_size
    end_y = start_y + box_size

    frame_cropped = frame[start_y:end_y, start_x:end_x]
    

    pil_image = Image.fromarray(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
    input_image = transform(pil_image)
    input_image = input_image.unsqueeze(0)

    with torch.no_grad():
        output = model(input_image)
        
    _, predicted_class = torch.max(output, 1)
    class_names = ['rock', 'scissor', 'paper']
    class_label = class_names[predicted_class.item()]

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    cv2.putText(frame, f"Class: {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Classification', frame)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()