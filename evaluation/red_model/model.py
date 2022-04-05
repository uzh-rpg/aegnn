import numpy as np
import torch

from torch.nn.functional import relu, sigmoid
from typing import Tuple

from evaluation.red_model.conv_lstm import ConvLSTM


class REDModel(torch.nn.Module):

    def __init__(self, in_channels: int, img_shape: Tuple[int, int]):
        super(REDModel, self).__init__()
        input_shape = np.array(img_shape)

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.sq2 = SqueezeExcitationLayer(32, 64, tuple(input_shape // 2))
        self.sq3 = SqueezeExcitationLayer(64, 64, tuple(input_shape // 8))

        self.conv4 = ConvLSTM(64, 256, kernel_size=(3, 3), stride=(1, 1), num_layers=1, return_all_layers=False)
        self.conv5 = ConvLSTM(256, 256, kernel_size=(3, 3), stride=(1, 1), num_layers=1)
        self.conv6 = ConvLSTM(256, 256, kernel_size=(3, 3), stride=(1, 1), num_layers=1)
        self.conv7 = ConvLSTM(256, 256, kernel_size=(3, 3), stride=(1, 1), num_layers=1)
        self.conv8 = ConvLSTM(256, 256, kernel_size=(3, 3), stride=(1, 1), num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = relu(self.conv1(self.bn1(x)))
        x = self.sq2(x)
        x = self.sq3(x)

        x = x.unsqueeze(dim=1)
        x = self.conv4(x)[0][-1]
        x = self.conv5(x)[0][-1]
        x = self.conv6(x)[0][-1]
        x = self.conv7(x)[0][-1]
        x = self.conv8(x)[0][-1]
        return x


class SqueezeExcitationLayer(torch.nn.Module):
    """Squeeze-and-Excitation Networks (https://arxiv.org/pdf/1709.01507.pdf)"""

    def __init__(self, in_channels: int, out_channels: int, input_shape: Tuple[int, int]):
        super(SqueezeExcitationLayer, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1)

        input_shape = tuple(np.ceil(np.ceil(input_shape) / 4).astype(int))
        self.pool4 = torch.nn.AvgPool2d(input_shape)  # M x N -> (1, 1)
        self.linear5 = torch.nn.Linear(out_channels, out_channels)
        self.linear6 = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = relu(self.conv1(self.bn1(x)))
        x = relu(self.conv2(self.bn2(x)))
        x = self.conv3(self.bn3(x))

        x_left = x.clone()
        _, _, w, h = x.shape

        x = self.pool4(x)
        x = torch.transpose(x, -3, -1)
        x = relu(self.linear5(x))
        x = torch.sigmoid(self.linear6(x))
        x = torch.transpose(x, -3, -1)

        x = x.expand(-1, -1, w, h)
        return x_left * (x + 1)
