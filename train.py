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

BATCH_SIZE = 64
MEMORY_SIZE = 1000
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
LR = .001
TARGET_UPDATE = 10

device = torch.device("cpu")
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

scores = []
steps = []

N_EPOCHS = 1000

# store transitions for each step to replay afterwards
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


policy_net = BasicLinearModel(BOARD_SIZE[0]+2, BOARD_SIZE[1]+2, 4).to(device)
target_net = BasicLinearModel(BOARD_SIZE[0]+2, BOARD_SIZE[1]+2, 4).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


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

def select_action(test_net, state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

def train_epoch(test1, test2, optimizer, memory):
    # check if batch can be constructed
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # create mask to filter final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # create batches
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # generate actions
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
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


    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)

    steps_done = 0

    for i in tqdm(range(n_epochs)):
        # create new game
        game = Snake(BOARD_SIZE[0], BOARD_SIZE[1], render=True)
        
        # get board states
        last_screen = torch.from_numpy(game.get_board_as_numpy(add_border=True, dtype=np.float32)).unsqueeze(0)
        current_screen = torch.from_numpy(game.get_board_as_numpy(add_border=True, dtype=np.float32)).unsqueeze(0)

        # state = current_screen - last_screen
        state = current_screen
        for t in count():
            # select an action
            action = select_action(policy_net, state, steps_done)
            steps_done += 1
            old_score = game.get_score()
            game.move(action.item())
            new_score = game.get_score()

            reward = 1
            if game.state == "failed":
                reward = -100
            elif new_score > old_score:
                reward = 10 * game.get_score()
            reward = torch.tensor([reward], device=device)
            done = True if game.state == "failed" else False

            # Observe new state
            last_screen = current_screen
            current_screen = torch.from_numpy(game.get_board_as_numpy(add_border=True,dtype=np.float32)).unsqueeze(0)
            if not done:
                # next_state = current_screen - last_screen
                next_state = current_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            train_epoch(policy_net,target_net,optimizer,memory)
            # print(policy_net.linear1.weight.sum())
            if done:
                scores.append(game.get_score())
                steps.append(t)
                ax1.clear()
                ax2.clear()
                ax1.plot(scores, "r", label="Scores", alpha=.5)
                ax2.plot(steps, "b", label="Steps", alpha=.5)
                fig.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                # print(t)
                # print(game.get_score())
                break

        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

train(N_EPOCHS)