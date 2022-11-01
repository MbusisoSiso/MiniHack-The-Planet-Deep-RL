import math
import os
import random
import fpdf
import gym
import matplotlib.pyplot as plt
import minihack
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from nle import nethack
from PIL import Image
from torch import flatten
from torch.nn import (BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Linear,MaxPool2d, Module, ReLU, Sequential, Softmax)
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pdf = fpdf.FPDF(format = 'letter')
pdf.add_page()
pdf.set_font('Arial', size = 9)
pdf.write(10, 'Robotics Assignment Logger')
pdf.ln()

def movingAverage(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

# Save state as image
def displayScreen(state, name):
    f = plt.figure()
    screen = Image.fromarray(np.uint8(state['pixel']))
    plt.imshow(screen)
    f.savefig(name)

# Obtained saved scores from csv file
def readResults(path):
    df = pd.read_csv(path)
    results = []
    for i in range(5):
        results.append(df.loc[i].to_list())
    return results

# Format state so neural network can accept
def formatState(state):
    glyphs = state["glyphs"]
    glyphs = glyphs/glyphs.max()
    glyphs = glyphs.reshape((1,1,21,79))
    return torch.from_numpy(glyphs).squeeze(0)

# Save plots of rewards per timestep for each environment averaged over iterations
def plotResults(env_name,scores, name, color='blue'):

    f = plt.figure(figsize=(8,6))   
    # Plot mean over all iterations
    mean = np.mean(scores,axis=0)
    plt.plot(mean, color=color,label="Mean Reward")
    plt.title(f"DQN - {env_name}")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
    f.savefig(name)


# Store transistions
class ReplayBuffer:

    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
 
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)

class DQN(nn.Module):

    def __init__(self, action_space: spaces.Discrete):

        super().__init__()
        
        self.conv1 = Conv2d(in_channels=1, out_channels=20,
            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=1600, out_features=500)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500, out_features=action_space.n)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# Define class that brings all the DQN compenents together so that a model can be trained
class DQNAgent():
    def __init__(self, observation_space, action_space, **kwargs):
        global device
        self.action_space = action_space
        self.replay_buffer = kwargs.get("replay_buffer", None)
        self.use_double_dqn = kwargs.get("use_double_dqn", None)
        self.gamma = kwargs.get("gamma", 0.99)
        self.lr = kwargs.get("lr", 1e-4)
        self.betas = kwargs.get("betas", (0.9, 0.999))
        self.batch_size = kwargs.get("batch_size", 256)
        # Create the online and target network
        self.online_network = DQN(action_space).to(device)
        self.target_network = DQN(action_space).to(device)
        self.optimiser = torch.optim.Adam(self.online_network.parameters(), lr=self.lr, betas=self.betas)
   

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = np.array(states)
        next_states = np.array(next_states)
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            if self.use_double_dqn:
                _, max_next_action = self.online_network(next_states).max(1)
                max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.online_network(next_states)
                max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values = self.target_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.online_network.state_dict())

    def act(self, observation):
        """Select action base on network inference"""
        if not torch.cuda.is_available():
            observation = observation.type(torch.FloatTensor) 
        else:
            observation = observation.type(torch.cuda.FloatTensor) 
        state = torch.unsqueeze(observation, 0).to(device)
        result = self.online_network.forward(state)
        action = torch.argmax(result).item()
        return action



def dqn(env, seed, learning_rate, max_episodes, max_episode_length, gamma, verbose=True):
    """
    Method to train DQN model.
    
    Input:
    env: The environment to be used during training
    seed: The random seed for any random operations performed 
    learning_rate: The learning rate uesd for the Adam optimizer when training the model 
    number_episodes: Number of episodes to train for 
    max_episode_length: The maximum number of steps to take in an episode before terminating
    gamma: The discount factor used when calculating the discounted rewards of an episode
    verbose: Print episode reward after each episode
    
    Returns:
    scores: The cumulative reward achieved by the agent for each episode during traiing
    """

    hyper_params = {
        'replay-buffer-size': int(1e6),
        'learning-rate': 0.01,
        'gamma': 0.99,  # discount factor
        'num-steps': int(2e5),  # Steps to run for, max episodes should be hit before this
        'batch-size': 32,  
        'learning-starts': 1000,  # set learning to start after 1000 steps of exploration
        'learning-freq': 1,  # Optimize after each step
        'use-double-dqn': True,
        'target-update-freq': 1000, # number of iterations between every target network update
        'eps-start': 1.0,  # e-greedy start threshold 
        'eps-end': 0.1,  # e-greedy end threshold 
        'eps-fraction': 0.4,  # Percentage of the time that epsilon is annealed
        'print-freq': 10,

    }
    
    np.random.seed(seed)
    env.seed(seed)
    
    # Create DQN agent
    replay_buffer = ReplayBuffer(hyper_params['replay-buffer-size'])
    agent = DQNAgent(
        env.observation_space, 
        env.action_space,
        train=True,
        replay_buffer=replay_buffer,
        use_double_dqn=hyper_params['use-double-dqn'],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        gamma=hyper_params['gamma'],
    )
    
    # define variables to track agent metrics
    total_reward = 0
    scores = []
    mean_rewards = []

    # Reset gym env before training
    state = formatState(env.reset())
    eps_timesteps = hyper_params['eps-fraction'] * float(hyper_params['num-steps'])
    # Train for set number of steps
    for t in range(hyper_params['num-steps']):
        # determine exploration probability
        fract = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fract * (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()
        # Decide to explore and choose random action or use model to act
        if sample < eps_threshold:
            action = np.random.choice(agent.action_space.n)
        else:
            action = agent.act(state)
        # Take step in environment
        (next_state, reward, done, _) = env.step(action)
        next_state = formatState(next_state)
        replay_buffer.add(state, action, reward, next_state, float(done))
        total_reward += reward
        state = next_state
        if done:
            scores.append(total_reward)
            print(f"episode reward: {total_reward}")
            np.random.seed(seed)
            env.seed(seed)
            state = formatState(env.reset())
            total_reward = 0

        if t > hyper_params['learning-starts'] and t % hyper_params['learning-freq'] == 0:
            ans = agent.optimise_td_loss()

        if t > hyper_params['learning-starts'] and t % hyper_params['target-update-freq'] == 0:
            agent.update_target_network()

        num_episodes = len(scores)
        if done and hyper_params['print-freq'] is not None and len(scores) % hyper_params['print-freq'] == 0:
            mean_100ep_reward = round(np.mean(scores[-101:-1]), 1)
            mean_rewards.append(mean_100ep_reward)
            print('********************************************************')
            print('steps: {}'.format(t))
            print('episodes: {}'.format(num_episodes))
            print('mean 100 episode reward: {}'.format(mean_100ep_reward))
            print('% time spent exploring: {}'.format(eps_threshold))
            print('********************************************************')
            pdf.write(10, '********************************************************')
            pdf.ln()
            pdf.write(10, 'steps: {}'.format(t))
            pdf.ln()
            pdf.write(10, 'episodes: {}'.format(num_episodes))
            pdf.ln()
            pdf.write(10, 'mean 100 episode reward: {}'.format(mean_100ep_reward))
            pdf.ln()
            pdf.write(10, '% time spent exploring: {}'.format(eps_threshold))
            pdf.ln()
            pdf.write(10, '********************************************************')
            pdf.ln()

        if num_episodes >=max_episodes:
            return scores

    return scores



def run_dqn(env,number_episodes,max_episode_length,iterations, name):
    """Trains DQN model for a number of episodes on a given environment"""
    seeds = np.random.randint(1000, size=iterations)
    scores_arr = [] 
    f = open(name, "w")
    writer = csv.writer(f)
    for seed in seeds:
        print(seed)
        # Train the DQN Model 
        scores = dqn(env=env, 
                            seed=seed, 
                            learning_rate=0.01,
                            max_episodes=1000, 
                            max_episode_length=1000, 
                            gamma=0.99 ,
                            verbose=True)
        # Store rewards for this iteration 
        scores_arr.append(scores)
    writer.writerow(scores_arr)
    f.close()
    return scores_arr


    # Create the environment with the observations keys required as input to the DQN
MOVE_ACTIONS = tuple(nethack.CompassDirection)
env = gym.make("MiniHack-Room-5x5-v0", observation_keys=["glyphs","pixel","message"], actions=MOVE_ACTIONS)
# Reset the environment and display the screen of the starting state 
displayScreen(env.reset(), 'Room-5x5')


# Train DQN on room 5x5
pdf.write(10, 'Room 5x5 Env')
pdf.ln()


room_5x5_scores = run_dqn(env,number_episodes=500,max_episode_length=200,iterations=5, name = "room_5x5_rewards.csv")
# The results from a training iteration have been saved and the output of this cell is blank to keep the notebook clean
# Please run cell if you want to train your own model

#room_5x5_scores = read_results("room_5x5_rewards.csv")

plotResults(env_name="Room-5x5-v0", scores=room_5x5_scores, color = "maroon", name = 'Room-5x5-v0.png')

# Create the environment with the observations keys required as input to the DQN
MOVE_ACTIONS = tuple(nethack.CompassDirection)
MOVE_ACTIONS += (nethack.Command.EAT,)
env = gym.make("MiniHack-Eat-v0", observation_keys=["glyphs","pixel","message"], actions=MOVE_ACTIONS)
# Reset the environment and display the screen of the starting state 
displayScreen(env.reset(), 'Eat')


pdf.write(10, 'Eat Env')
pdf.ln()
# Train DQN on eat
eat_scores = run_dqn(env,number_episodes=500,max_episode_length=200,iterations=5, name = "eat_rewards.csv")
# The results from a training iteration have been saved and the output of this cell is blank to keep the notebook clean
# Please run cell if you want to train your own model

#eat_scores = read_results("eat_rewards.csv")

plotResults(env_name="Eat", scores=eat_scores, color = "blue", name = 'Eat.png')


# Create the environment with the observations keys required as input to the DQN
MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = MOVE_ACTIONS + (
    nethack.Command.PICKUP,
    nethack.Command.APPLY,
    nethack.Command.FIRE,
    nethack.Command.RUSH,
    nethack.Command.ZAP,
    nethack.Command.PUTON,
    nethack.Command.READ,
    nethack.Command.WEAR,
    nethack.Command.QUAFF,
    nethack.Command.PRAY,
    )
env = gym.make("MiniHack-Quest-Hard-v0", observation_keys=["glyphs","pixel","message"], actions=NAVIGATE_ACTIONS)
# Reset the environment and display the screen of the starting state 
displayScreen(env.reset(), 'Quest-Hard')



pdf.write(10, 'Quest-Hard Env')
pdf.ln()
# Train DQN on Quest Hard
quest_hard_scores = run_dqn(env,number_episodes=500,max_episode_length=200,iterations=5, name = "quest_hard_rewards.csv")
# The results from a training iteration have been saved and the output of this cell is blank to keep the notebook clean
# Please run cell if you want to train your own model

#quest_hard_scores = read_results("quest_hard_rewards.csv")

plotResults(env_name="Quest-Hard", scores=quest_hard_scores, color = "yellow", name = 'Quest-Hard.png')

pdf.output('DQNOutput.pdf')