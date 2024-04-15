import torch
import torch.nn as nn
import torch.optim as optim

class KeyPointsCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KeyPointsCNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (input_dim // 4), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * (self.input_dim // 4))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
input_dim = 20  # Dimension of the input sequence (number of keypoints * 2)
hidden_dim = 128
output_dim = 3  # Number of classes (起跳、落地、跌倒)

# Create an instance of the model
model = KeyPointsCNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
