from tqdm import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

import numpy as np 
import gym 
import random 
import time 

from model import QNetwork 
from utils import TensorEnv, ExperienceReplayMemory, AgentConfig
import utils

args = utils.arguments()


env = TensorEnv(args.env)
        
agent = QNetwork(env.observation_space.shape[0], env.action_space.n)
agent.load_state_dict(torch.load('./runs/{}/model_state_dict'.format(args.env)))

for ep in tqdm(range(10)):

    s = env.reset()
    done = False
    while not done: 

        action = agent(s).reshape(-1).argmax().item()

        s, r, done, _ = env.step(action)

        env.render()
        if not args.env.startswith('Lunar'): 
           time.sleep(0.02) 

env.close()