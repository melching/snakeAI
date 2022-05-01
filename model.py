import torch
import torch.nn as nn

class BasicLinearModel(nn.Module):
    def __init__(self, x_dim, y_dim, value_dim, output_dim) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.value_dim = value_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(self.x_dim*self.y_dim*self.value_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear_out = nn.Linear(256 + 4, self.output_dim) # +4 due to danger

        self.activation = nn.LeakyReLU(negative_slope=.01)
        self.activation_out = nn.Softmax(dim=0)

    def forward(self, x, danger):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        # x = self.activation(x)
        x = self.linear2(x)
        # x = self.activation(x)
        out = self.linear_out(torch.cat([x, danger], dim=1))
        # x = self.activation_out(x)

        return out

class BasicConvModel(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(4, 32, 3, padding="same")
        self.conv2 = nn.Conv2d(32, 16, 3, padding="same")
        self.linear1 = nn.Linear(x_dim * y_dim * 16, 64)
        self.linear_out = nn.Linear(64 + 4, output_dim)

    def forward(self, x, danger):
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear_out(torch.cat([x, danger], dim=1))
        return x