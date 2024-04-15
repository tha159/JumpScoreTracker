import torch
import json
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import json
import time

import cv2

def draw_skeleton(image, frame_data):
    # print(frame_data)
    # 定义连接关系
    connections = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (0, 6), (1, 7), (6, 8), (8, 10), (10, 12), (12, 14), (1, 7),
                   (7, 9), (9, 11), (11, 13), (13, 15)]

    # 绘制关键点


    for i in range(len(frame_data)):
        key_point = frame_data[i]
        x, y = key_point[1], key_point[2]
        cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)  # 画关键点

    # 绘制骨骼
    for connection in connections:
        start_point = frame_data[connection[0]][1:]  # 获取起始点坐标
        end_point = frame_data[connection[1]][1:]  # 获取结束点坐标
        cv2.line(image, tuple(start_point), tuple(end_point), (0, 0, 255), 2)

    return image
# 定义模型类
class KeyPointsLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KeyPointsLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# 准备数据集类
class MyDataset(Dataset):
    def __init__(self, data_path):
        key_points_list = []
        with open(data_path, "r") as file:
            for line in file:
                key_points_data = json.loads(line)
                key_points_list.append(key_points_data)
        self.data = key_points_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key_points, label = self.data[idx]
        key_points_tensor = torch.tensor([[point[0]] + point[1:] for point in key_points], dtype=torch.float32)
        return key_points_tensor, label

# 加载模型
input_dim = 3
hidden_dim = 128
output_dim = 4
model = KeyPointsLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model.load_state_dict(torch.load("keypoints_lstm_model2.pth"))
model.eval()

# 读取文件
file_path = "data_train.txt"
key_points_list = []
with open(file_path, "r") as file:
    for line in file:
        # 解析每个JSON对象
        key_points_data = json.loads(line)
        key_points_list.append(key_points_data)

frame_index = 0
while True:
    # 获取当前帧数据
    frame_data = key_points_list[frame_index][0]

    # 构造图像
    # 这里你需要自己写代码来绘制关键点
    image = cv2.imread(r'D:\user\work\sport\sport_demo1\img\2findRuler.jpg')

    # 根据 frame_data 绘制图像
    image = draw_skeleton(image, frame_data)

    # 准备输入数据并进行预测
    input_data = torch.tensor([frame_data], dtype=torch.float32)  # 将当前帧数据转换为张量
    with torch.no_grad():
        output = model(input_data)  # 使用模型进行预测

    _, predicted_label = torch.max(output, 1)  # 找到预测结果中概率最大的类别
    predicted_label = predicted_label.item()  # 将张量转换为标量

    cv2.putText(image, str(predicted_label), (125,55), cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                4)

    # 显示图像
    cv2.imshow('Frame', image)
    # 监听键盘事件
    key = cv2.waitKey(0) & 0xFF

    # 处理键盘事件
    if key == ord('s'):  # 保存但不退出
        pass
    elif key == ord('a'):  # 方向键 ←
        frame_index -= 1
        if frame_index < 0:
            frame_index = 0
        print('previous frame')
    elif key == ord('d'):  # 方向键 →
        frame_index += 1
        if frame_index >= len(key_points_list):
            frame_index = len(key_points_list) - 1
        print('next frame')

    # 检查是否按下了退出键
    if key == ord('q'):
        break

cv2.destroyAllWindows()


