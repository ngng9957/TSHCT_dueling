#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import random
import os
import sys

import math
import numpy as np

import matplotlib.pyplot as plt

import time
import pickle

data_path = os.path.join(os.path.dirname(__file__), 'REWARD_DRQN.pickle')
data_path_du = os.path.join(os.path.dirname(__file__), 'REWARD_Dueling_DRQN.pickle')

class Plot():
    def __init__(self):
        self.episode = []
        self.m_episode = []
        self.num_value = 5
        self.mean_value = [[] for _ in range(self.num_value)]
        self.mean_value_du = [[] for _ in range(self.num_value)]
        self.upper_value = [[] for _ in range(self.num_value)]
        self.lower_value = [[] for _ in range(self.num_value)]
        self.upper_value_du = [[] for _ in range(self.num_value)]
        self.lower_value_du = [[] for _ in range(self.num_value)]
        self.init_value = [0,0,0,0,0]
        self.init_value_du = [0,0,0,0,0]

        with open(data_path,"rb") as fr:
            self.value = pickle.load(fr)
        with open(data_path_du,"rb") as fr:
            self.value_du = pickle.load(fr)
        self.num = 10
        self.num_2 = 100

        print("Initializing variables...")

    def run(self):
        # for i in range(10000):
        for i in range(2000):
            self.episode.append(100*(i+1))
            if i >= self.num:
                if self.num == i:
                    for role in range(self.num_value):
                        self.init_value[role] = np.mean(self.value[role][i-self.num:i])
                        self.init_value_du[role] = np.mean(self.value_du[role][i-self.num:i])
                self.m_episode.append(self.episode[i] - self.num/2)
                for role in range(self.num_value):
                    data = -self.init_value[role] + np.mean(self.value[role][i-self.num:i])
                    self.mean_value[role].append(data)
                    data_du = -self.init_value_du[role] + np.mean(self.value_du[role][i-self.num:i])
                    self.mean_value_du[role].append(data_du)
                    if i >= self.num_2:
                        self.upper_value[role].append(  max(self.value[role][i-self.num_2:i])-self.init_value[role])
                        self.lower_value[role].append(  min(self.value[role][i-self.num_2:i])-self.init_value[role])
                        self.upper_value_du[role].append(  max(self.value_du[role][i-self.num_2:i])-self.init_value_du[role])
                        self.lower_value_du[role].append(  min(self.value_du[role][i-self.num_2:i])-self.init_value_du[role])
        xlab = 'Training Episodes'
        ylab = 'Increased Reward from Initial Value'
        color='r'
        color_du = 'indigo'
        label='TSHCT-DRQN'
        label_du='TSHCT-Dueling DRQN'
        names = ['Goalkeeper Reward',
                 'Defenders Reward',
                 'Forwards Reward',
                 'Team Reward',
                 'Total Reward'
                ]
        for i in range(5):
            if i == 4:
                plt.title(names[i])
                plt.xlabel(xlab)
                plt.ylabel(ylab)
            else:
                plt.title(names[i], size = 22)
            plt.xticks([ 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000, 380000, 400000],
             ['20k', '40k', '60k', '80k', '100k', '120k', '140k', '160k', '180k', '200k', '220k', '240k', '260k', '280k', '300k', '320k', '340k', '360k', '380k', '400k'])
            plt.ylim( -0.1, max(self.mean_value[i]+self.mean_value_du[i])+0.1)
            plt.plot(self.m_episode, self.mean_value[i], c = color, label= label) 
            plt.plot(self.m_episode, self.mean_value_du[i], c = color_du, label= label_du)
            plt.grid(True)
            if i == 4:
                plt.legend(loc=2)
            fig_path = os.path.join(os.path.dirname(__file__), names[i]+'.png')
            plt.savefig(fig_path, format='png')
            plt.close()
            print("plot", i)

        print("plot all")
        

if __name__ == '__main__':
    main = Plot()
    main.run()