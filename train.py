'''
Script according to https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

import torch
import torch.nn as nn
import numpy as np
import math
from model import BasicLinearModel, BasicConvModel

import time
import random
from tqdm import tqdm
from collections import namedtuple, deque
from itertools import count
import matplotlib.pyplot as plt

from snake import Snake

# Game settings
BOARD_SIZE = (10, 10)
ADD_BORDER = False

PLOT_EVERY = 100

BATCH_SIZE = 512
MEMORY_SIZE = 10000
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400
LR = .01
TARGET_UPDATE = 10

N_EPOCHS = 1000

device = torch.device("cpu")
if PLOT_EVERY:
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

scores = []
steps = []



# store transitions for each step to replay afterwards
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(policy_net, state, danger, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state, danger).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

def train_epoch(policy_net, target_net, optimizer, memory):
    # check if batch can be constructed
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # create mask to filter final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s[0] is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states  = torch.cat([s[0] for s in batch.next_state if s[0] is not None])
    non_final_next_dangers = torch.cat([s[1] for s in batch.next_state if s[0] is not None])

    # create batches
    state_batch  = torch.cat([s[0] for s in batch.state])
    danger_batch = torch.cat([s[1] for s in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # generate actions
    state_action_values = policy_net(state_batch, danger_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states, non_final_next_dangers).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(loss.item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train(n_epochs):
    # init models and memory
    border = 2 if ADD_BORDER else 0
    policy_net = BasicLinearModel(BOARD_SIZE[0]+border, BOARD_SIZE[1]+border, 4).to(device)
    target_net = BasicLinearModel(BOARD_SIZE[0]+border, BOARD_SIZE[1]+border, 4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0

    for i in tqdm(range(n_epochs)):
        # create new game
        game = Snake(BOARD_SIZE[0], BOARD_SIZE[1], render=True)
        
        # get board state
        current_state, current_danger = game.get_board_as_numpy(add_border=ADD_BORDER, add_danger=True, as_cat=True, dtype=np.float32)
        current_state, current_danger = torch.from_numpy(current_state).unsqueeze(0), torch.from_numpy(current_danger).unsqueeze(0)

        for t in count():
            # select an action
            action = select_action(policy_net, current_state, current_danger, steps_done)
            steps_done += 1

            current_score = game.get_score()
            game.move(action.item())
            new_score = game.get_score()

            reward = (BOARD_SIZE[0] + BOARD_SIZE[1]) - (abs(game.snake_body[0][0] - game.apple[0]) + abs(game.snake_body[0][1] - game.apple[1]))
            if game.state == "failed":
                reward = -1000
            elif new_score > current_score:
                reward = 10 * game.get_score()
            reward = torch.tensor([reward], device=device)
            done = True if game.state == "failed" else False

            # Observe new state
            next_state, next_danger = game.get_board_as_numpy(add_border=ADD_BORDER, add_danger=True, as_cat=True, dtype=np.float32)
            next_state, next_danger = torch.from_numpy(next_state).unsqueeze(0), torch.from_numpy(next_danger).unsqueeze(0)
            if done:
                next_state = None

            # Store the transition in memory
            memory.push((current_state, current_danger), action, (next_state, next_danger), reward)

            # Move to the next state
            current_state = next_state

            # Perform one step of the optimization (on the policy network)
            train_epoch(policy_net, target_net, optimizer, memory)

            # plot if done
            if done:
                scores.append(game.get_score())
                steps.append(t)
                break

        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if PLOT_EVERY and i % PLOT_EVERY == 0:
            ax1.clear()
            ax2.clear()
            ax1.plot(scores, "r", label="Scores", alpha=.5)
            ax2.plot(steps, "b", label="Steps", alpha=.5)
            fig.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()

            print("Frames Seen", steps_done)

train(N_EPOCHS)