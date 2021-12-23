from model import BasicLinearModel
from snake import Snake
import torch
import numpy as np

game = Snake(20,20)
model = BasicLinearModel(20,20,4)

state = game.get_board_as_numpy(dtype=np.float32)
state = torch.from_numpy(state).unsqueeze(dim=0)

print(model(state))
