import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from scipy.special import gamma  # 引入伽马函数

class CaputoEncoder(nn.Module):
    def __init__(self, input_size=250, lstm_size=512, lstm_layers=2, output_size=1024, alphas=[0.5, 1.0]):
        super(CaputoEncoder, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.alphas = alphas

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size * len(alphas), lstm_size, num_layers=lstm_layers, batch_first=True)

        # 定义输出层
        self.output = nn.Linear(lstm_size, output_size)

    def caputo_derivative(self, x, alpha):
        """
        计算Caputo分数阶导数（矢量化版本）
        :param x: 输入信号, 形状为 (batch_size, time_steps, channels)
        :param alpha: 分数阶阶数
        :return: 分数阶导数特征
        """
        batch_size, T, N = x.shape
        D_alpha_X = torch.zeros_like(x)  # 初始化分数阶导数结果

        # 计算系数
        coef = 1 / gamma(1 - alpha)

        # 计算时间差矩阵 (T x T)，使用广播
        time_indices = torch.arange(T).float().to(x.device)
        time_diff = (time_indices.view(-1, 1) - time_indices.view(1, -1)).abs() + 1e-6
        time_diff = torch.tril(time_diff, diagonal=-1) ** (-alpha)  # 只取下三角部分
        
        # 使用广播机制进行批量计算
        for t in range(1, T):
            diffs = x[:, t, :].unsqueeze(1) - x[:, :t, :]  # 计算时间步差异
            weights = time_diff[t, :t].view(1, -1, 1)  # 权重矩阵
            D_alpha_X[:, t, :] = coef * torch.sum(diffs * weights, dim=1)  # 计算加权和

        return D_alpha_X

    def forward(self, x):
        # 将输入信号转换为浮点型张量
        x = x.float()

        # 计算多尺度分数阶导数特征
        features = []
        for alpha in self.alphas:
            D_alpha_X = self.caputo_derivative(x, alpha)
            features.append(D_alpha_X)
        
        # 将多尺度特征拼接
        x = torch.cat(features, dim=2)

        # 将拼接后的特征传递给LSTM层
        lstm_init = (torch.zeros(self.lstm_layers, x.size(0), self.lstm_size).to(x.device),
                     torch.zeros(self.lstm_layers, x.size(0), self.lstm_size).to(x.device))
        x, _ = self.lstm(x, lstm_init)

        # 选取最后一个时间步的输出
        x = x[:, -1, :]

        # 通过全连接层
        x = F.relu(self.output(x))

        return x



# Test the FreqEncoder model
def test_model():
    # Instantiate the model and move it to GPU
    model = CaputoEncoder(input_size=250, lstm_size=512, lstm_layers=2, output_size=1024)
    model = model.cuda()

    # Define a random input tensor with shape (batch_size, 63, 250)
    batch_size = 256
    sequence_length = 63
    input_size = 250
    input_tensor = torch.randn(batch_size, sequence_length, input_size).cuda()

    # Forward pass through the model
    output = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)  # Expected: (batch_size, 1024)

    # Verify the output dimensions
    assert output.shape == (batch_size, 1024), "Output shape is incorrect!"

    print("Test passed!")

# Run the test
if __name__ == "__main__":
    test_model()
