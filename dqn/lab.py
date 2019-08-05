"""DLP DQN Lab"""
__author__ = 'chengscott'
__copyright__ = 'Copyright 2019, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import torch
import torch.nn as nn


class ReplayMemory:
  def __init__(self, capacity):
    self._buffer = deque(maxlen=capacity)

  def __len__(self):
    return len(self._buffer)

  def append(self, *transition):
    # (state, action, reward, next_state, done)
    self._buffer.append(tuple(map(tuple, transition)))

  def sample(self, batch_size=1):
    return random.sample(self._buffer, batch_size)


class DQN(nn.Module):
  def __init__(self, state_dim=4, action_dim=2, hidden_dim=24):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
    )

  def forward(self, x):
    return self.layers(x)


def select_action(epsilon, state, action_dim=2):
  """epsilon-greedy based on behavior network"""
  if random.random() < epsilon:
    return random.randrange(action_dim)
  with torch.no_grad():
    return behavior_net(state).argmax().item()


def update_behavior_network():
  def transitions_to_tensors(transitions, device=args.device):
    """convert a batch of transitions to tensors"""
    return (torch.Tensor(x).to(device) for x in zip(*transitions))

  # sample a minibatch of transitions
  transitions = memory.sample(args.batch_size)
  state, action, reward, next_state, done = transitions_to_tensors(transitions)
  # loss
  q_value = behavior_net(state).gather(1, action.long())
  with torch.no_grad():
    q_next = target_net(next_state).max(dim=1).values.unsqueeze(1)
    q_next = (1 - done) * args.gamma * q_next + reward
  loss = criterion(q_value, q_next)
  # optimize
  optimizer.zero_grad()
  loss.backward()
  nn.utils.clip_grad_norm_(behavior_net.parameters(), 5)
  optimizer.step()


def train(env):
  print('Start Training')
  total_steps, epsilon = 0, 1.
  for episode in range(args.episode):
    total_reward = 0
    state = env.reset()
    for t in itertools.count(start=1):
      # select action
      if total_steps < args.warmup:
        action = env.action_space.sample()
      else:
        state_tensor = torch.Tensor(state).to(args.device)
        action = select_action(epsilon, state_tensor)
        epsilon = max(epsilon * args.eps_decay, args.eps_min)
      # execute action
      next_state, reward, done, _ = env.step(action)
      # store transition
      memory.append(state, [action], [reward / 10], next_state, [int(done)])
      if total_steps >= args.warmup and total_steps % args.freq == 0:
        # update the behavior network
        update_behavior_network()
      if total_steps % args.target_freq == 0:
        # update the target network by copying from the behavior network
        target_net.load_state_dict(behavior_net.state_dict())

      state = next_state
      total_reward += reward
      total_steps += 1
      if done:
        print('Step: {}\tEpisode: {}\tTotal reward: {}\tEpsilon: {}'.format(
            total_steps, episode, total_reward, epsilon))
        break
  env.close()


def test(env, render):
  print('Start Testing')
  epsilon = args.test_epsilon
  seeds = (856032 + i for i in range(10))
  for seed in seeds:
    total_reward = 0
    env.seed(seed)
    state = env.reset()
    while True:
      state_tensor = torch.Tensor(state).to(args.device)
      action = select_action(epsilon, state_tensor)
      if render:
        env.render()
        time.sleep(0.01)
      next_state, reward, done, _ = env.step(action)
      state = next_state
      total_reward += reward
      if done:
        print('Seed: {}\tTotal reward: {}'.format(seed, total_reward))
        break
  env.close()


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-d', '--device', default='cuda')
  # network
  parser.add_argument('-m', '--model', default='cartpole_model')
  parser.add_argument('--restore', action='store_true')
  # train
  parser.add_argument('-e', '--episode', default=1800, type=int)
  parser.add_argument('-c', '--capacity', default=10000, type=int)
  parser.add_argument('-bs', '--batch_size', default=128, type=int)
  parser.add_argument('--warmup', default=10000, type=int)
  parser.add_argument('--lr', default=.0005, type=float)
  parser.add_argument('--eps_decay', default=.995, type=float)
  parser.add_argument('--eps_min', default=.01, type=float)
  parser.add_argument('--gamma', default=.99, type=float)
  parser.add_argument('--freq', default=4, type=int)
  parser.add_argument('--target_freq', default=1000, type=int)
  # test
  parser.add_argument('--test_epsilon', default=.001, type=float)
  parser.add_argument('--render', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  # environment
  env = gym.make('CartPole-v1')
  # behavior network
  behavior_net = DQN().to(args.device)
  if not args.restore:
    # target network
    target_net = DQN().to(args.device)
    # initialize target network
    target_net.load_state_dict(behavior_net.state_dict())
    # optimizer
    optimizer = torch.optim.Adam(behavior_net.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    # memory
    memory = ReplayMemory(capacity=args.capacity)
    # train
    train(env)
    # save model
    torch.save(behavior_net, args.model)
  # load model
  behavior_net = torch.load(args.model)
  # test
  test(env, args.render)
