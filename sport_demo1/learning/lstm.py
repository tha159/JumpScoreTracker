import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils

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

class MyDataset(Dataset):
    def __init__(self, data_path):
        # 读取文件
        key_points_list = []

        with open(data_path, "r") as file:
            for line in file:
                # 解析每个JSON对象
                key_points_data = json.loads(line)
                key_points_list.append(key_points_data)

        print(len(key_points_list))
        self.data = key_points_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key_points, label = self.data[idx]
        # 转换关键点数据格式为张量
        key_points_tensor = torch.tensor([[point[0]] + point[1:] for point in key_points], dtype=torch.float32)
        # 返回输入数据、标签和关键点序号
        return key_points_tensor, label



# Example usage:
input_dim = 3  # Dimension of each key point (x, y, z)
hidden_dim = 128
output_dim = 4  # Number of classes (0为站，1为蹲，2为向上跳跃，3为向下落下，4为跌倒)

# 准备训练数据集
train_dataset = MyDataset("data_train2.txt")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 创建一个模型实例
model = KeyPointsLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始模型的训练
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# 保存模型
torch.save(model.state_dict(), "keypoints_lstm_model2.pth")
