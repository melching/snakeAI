import torch
import torch.nn as nn

class BasicLinearModel(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(self.x_dim*self.y_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear_out = nn.Linear(256, self.output_dim)

        self.activation = nn.LeakyReLU(negative_slope=.01)
        self.activation_out = nn.Softmax(dim=0)

    def forward(self,x):
        # x = x.transpose(1,2)
        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)
        # x = self.activation(x)
        x = self.linear2(x)
        # x = self.activation(x)
        x = self.linear_out(x)
        # x = self.activation_out(x)

        return x

class BasicConvModel(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 64, 3, padding="same")
        self.conv2 = nn.Conv2d(64, 32, 3, padding="same")
        self.linear_out = nn.Linear(x_dim * y_dim * 32, output_dim)

    def forward(self, x):
        x = x.transpose(1,2)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_out(x)
        return x