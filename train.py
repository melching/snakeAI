import torch
import numpy as np
import time
from snake import Snake

BOARD_SIZE = (40, 40)
MAX_ITERATION_WITHOUT_REWARD = 100


def train_epoch(model):
    # init new game
    game = Snake(BOARD_SIZE[0], BOARD_SIZE[1], render=True)
    game_iteration = 0
    score = game.get_score()

    while game.state == "running" and game_iteration < MAX_ITERATION_WITHOUT_REWARD:
        # get state and parse to model
        state = game.get_board_as_numpy(dtype=np.int)
        prediction = model(state)

        # process result and perform move
        direction = torch.argmax(prediction)
        game.move(direction)

        new_score = game.get_score()

    
    return score