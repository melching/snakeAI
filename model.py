import torch
import torch.nn as nn

class BasicLinearModel(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(self.x_dim*self.y_dim, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear_out = nn.Linear(1024, self.output_dim)

        self.activation = nn.LeakyReLU(negative_slope=.01)
        self.activation_out = nn.Softmax()

    def forward(self,x):
        x = torch.flatten(x)

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear_out(x)
        x = self.activation_out(x)

        return x