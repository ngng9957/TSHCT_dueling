#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import random
import os
import sys

import math
import numpy as np

import helper
from drqn import DRQN
from rl_utils import  Logger

import time
import torch
import pickle

# for webots
import asyncio
import subprocess


CHECKPOINT_GK = os.path.join(os.path.dirname(__file__), 'models/Robot_GK.th')
CHECKPOINT_D12 = os.path.join(os.path.dirname(__file__), 'models/Robot_D12.th')
CHECKPOINT_F12 = os.path.join(os.path.dirname(__file__), 'models/Robot_F12.th')
CHECKPOINT = [CHECKPOINT_GK, CHECKPOINT_D12, CHECKPOINT_F12]
CHECKPOINT_TEAM_MIXER =  os.path.join(os.path.dirname(__file__), 'models/team_mixer.th')

models_path = os.path.join(os.path.dirname(__file__), 'models/')
nets_name = ['/Robot_GK.th','/Robot_D12.th','/Robot_F12.th']
team_mixer_name = ['/team_mixer.th']

config_path = os.path.join(os.path.dirname(__file__), 'config.pickle')
memory_path = os.path.join(os.path.dirname(__file__), 'memory.pickle')

class TestPlayer():
    def __init__(self):
        self.number_of_robots = 5
        self.end_of_frame = False
        self.epi_max_len = 40
        self.obs_size = 30 
        self.state_size = 22
        self.act_size = 20
        self.role_type = 3
        self.t = 0
        # for RL
        self.action = [0 for _ in range(self.number_of_robots)]
        self.pre_action = [0 for _ in range(self.number_of_robots)]

        self.load = False
        self.epsilon = 0.95
        self.trainer = DRQN(self.number_of_robots, self.obs_size, self.state_size, self.act_size, self.epi_max_len, self.epsilon, self.load)
        self.episode = 0
        
        self.cur_memory = []
        self.cur_mtime = 0
        self.pre_mtime = 0

        self.plot_reward = Logger()
        self.plot_update = 100
        self.mean_reward = [[0 for _ in range(self.plot_update)] for _ in range(5)]
        self.reward_t = 0
        self.save_png_interval = 100
        self.cur_time = time.time()
        self.episode_plus_time = time.time()

        self.done = True
        with open(config_path,"wb") as fw:
            config_data = [self.load, self.trainer.epsilon, self.episode]
            pickle.dump(config_data, fw)

        helper.printConsole("Initializing variables...")

    def run(self):
        self.cur_time = time.time()
        if self.pre_mtime != self.cur_mtime:
            time.sleep(0.2)
            try:
                with open(memory_path,"rb") as fr:
                    self.cur_memory = pickle.load(fr)
                if self.episode % 10 == 0:  
                    with open(config_path,"wb") as fw:
                        config_data = [self.load, self.trainer.epsilon, self.episode]
                        pickle.dump(config_data, fw)
                self.trainer.memory.push(self.cur_memory[0],self.cur_memory[1],self.cur_memory[2],self.cur_memory[3],self.cur_memory[4],self.cur_memory[5],self.cur_memory[6])
                gk_reward=sum(self.cur_memory[4][:,0])/sum(self.cur_memory[6])
                d12_reward=sum(self.cur_memory[4][:,1])/sum(self.cur_memory[6])
                f12_reward=sum(self.cur_memory[4][:,2])/sum(self.cur_memory[6])
                team_reward=sum(self.cur_memory[5])/sum(self.cur_memory[6])
                total_reward =gk_reward + d12_reward + f12_reward + team_reward
                mean_reward = [round(gk_reward[0],2), round(d12_reward[0],2), round(f12_reward[0],2), round(team_reward[0],2), round(total_reward[0],2)]
                self.mean_reward[0][self.reward_t] = mean_reward[0]
                self.mean_reward[1][self.reward_t] = mean_reward[1]
                self.mean_reward[2][self.reward_t] = mean_reward[2]
                self.mean_reward[3][self.reward_t] = mean_reward[3]
                self.mean_reward[4][self.reward_t] = mean_reward[4]
                self.reward_t += 1
                print(mean_reward)

                if self.episode % self.plot_update == 0:
                    update_reward =[round(sum(self.mean_reward[num])/self.plot_update, 2) for num in range(5)] 
                    self.plot_reward.update(self.episode, update_reward, 10)
                    self.reward_t = 0
                    self.mean_reward = [[0 for _ in range(self.plot_update)] for _ in range(5)]
                if self.episode % self.save_png_interval == 0:
                    self.plot_reward.plot('MULTI-AGENT-REWARD')
                self.cur_memory = []
                self.pre_mtime = self.cur_mtime
                self.episode += 1
                self.done = False
                self.episode_plus_time = time.time()
                print("memory_len :",len(self.trainer.memory))
                print("iterations :",self.trainer._iterations)
            except:    
                print("loading err occur")
                pass

        if os.path.isfile(memory_path):
            self.cur_mtime = os.path.getmtime(memory_path)

        ## training part ##
        if (self.episode <= self.trainer.observation_steps) and (self.episode % 100== 0):
            for role in range(self.role_type):
                self.trainer.net[role].save_model(self.trainer.net[role], CHECKPOINT[role])
        elif self.episode > self.trainer.observation_steps:
            self.trainer.update(self.episode)
            if self.done == False:
                for role in range(self.role_type):
                    self.trainer.net[role].save_model(self.trainer.net[role], CHECKPOINT[role])
                if self.episode % 100 == 0:
                    torch.save(self.trainer.team_mixer.state_dict(),CHECKPOINT_TEAM_MIXER)
                if self.episode % 1000 == 0:
                    if os.path.isdir(models_path + str(self.episode) + '/') == False:
                        os.mkdir(models_path + str(self.episode) + '/')
                    for role in range(self.role_type):
                        path = models_path + str(self.episode) + nets_name[role]
                        self.trainer.net[role].save_model(self.trainer.net[role], path)
                    path = models_path + str(self.episode) + team_mixer_name[0]
                    torch.save(self.trainer.team_mixer.state_dict(),path)
                self.done = True

class webots():
    def __init__(self):
        self.done = True
        self.sims = subprocess.Popen('webots') 
        self.sims_state = True 
        print("Initializing variables...")

    def run(self):
        if self.sims_state == False:
            time.sleep(1)
            self.sims = subprocess.Popen('webots')
            print("WEBOTS RUN")
            self.sims_state = True
        if self.sims.poll() != None:
            self.sims_state = False



if __name__ == '__main__':
    player = TestPlayer()
    webots = webots()
    time.sleep(1)

    while True:
        episode_time_interval = player.cur_time - player.episode_plus_time
        webots.run()
        player.run()
        if episode_time_interval > 20:
            webots.sims.terminate()
            webots.sims_state = False
            player.episode_plus_time = time.time()