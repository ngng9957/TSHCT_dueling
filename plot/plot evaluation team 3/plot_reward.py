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

data_path1 = os.path.join(os.path.dirname(__file__), 'REWARD_TSHCT.pickle')
data_path2 = os.path.join(os.path.dirname(__file__), 'REWARD_QMIX.pickle')
data_path3 = os.path.join(os.path.dirname(__file__), 'REWARD_VDN.pickle')
data_path4 = os.path.join(os.path.dirname(__file__), 'REWARD_COMA.pickle')
GK = 0
D12 = 1
F12 = 2
TEAM = 3
TOTAL = 4
class Plot():
    def __init__(self, name):
        self.episode = []
        self.m_episode = []
        self.num_value = 5
        self.name = name
        self.mean_value =  []
        self.upper_value =  []
        self.lower_value =  []
        self.cur_value =  []
        self.init_value = 0


        with open(data_path1,"rb") as fr:
            self.value1 = pickle.load(fr)
        with open(data_path2,"rb") as fr:
            self.value2 = pickle.load(fr)
        with open(data_path3,"rb") as fr:
            self.value3 = pickle.load(fr)
        with open(data_path4,"rb") as fr:
            self.value4 = pickle.load(fr)
        self.cur_num = 2
        self.num = 6
        self.role = {'Goalkeeper':0, 'Defenders':1, 'Forwards':2, 'Team':3, 'Total':4}
        self.num = {'Goalkeeper':10, 'Defenders':10, 'Forwards':10, 'Team':10, 'Total':10}
        self.value = [self.value1[self.role[self.name]],self.value2[self.role[self.name]],self.value3[self.role[self.name]],self.value4[self.role[self.name]]]


        print("Initializing variables...")

    def run(self,name):
        plt.clf()
        for j in [0,3,2,1]:
        # for j in range(4):

            for i in range(200):
                self.episode.append(1000*(i+1))
                if i >= self.num[name]:    
                    self.m_episode.append(self.episode[i] - self.num[name]/2)
                    self.cur_value.append( np.mean(self.value[j][i-self.cur_num:i]))
                    # print(i,self.value[j][i-self.num:i])
                    self.mean_value.append( np.mean(self.value[j][i-self.num[name]:i]))
                    self.upper_value.append(  max(self.value[j][i-self.num[name]:i]))
                    self.lower_value.append(  min(self.value[j][i-self.num[name]:i]))

            if name == 'Total':
                plt.title(name + ' Reward')
                xlab = 'Episodes'
                ylab = str(name) +' Reward'
                plt.xlabel(xlab)
                plt.ylabel(ylab)
            else:
                plt.title(name + ' Reward',size=22)
            color=['pink','red','lightskyblue','b','lightgreen','green','moccasin','orange']
            # color=['lightskyblue','b','pink','deeppink','lightcoral','red']
            label=['TSHCT','QMIX','VDN','COMA']
            plt.xticks([ 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000, 380000, 400000],
             ['20k', '40k', '60k', '80k', '100k', '120k', '140k', '160k', '180k', '200k', '220k', '240k', '260k', '280k', '300k', '320k', '340k', '360k', '380k', '400k'])            # plt.ylim(-0.05,  10)

            if self.name =='Goalkeeper':
                plt.ylim(0.2,1.2)
            if self.name =='Defenders':
                plt.ylim(0.5,2)
            if self.name =='Forwards':
                plt.ylim(0.5,2)
            if self.name =='Team':
                plt.ylim(0.5,2)
            if self.name =='Total':
                plt.ylim(2,7)


            if j == 0:
                # plt.plot(self.m_episode, self.cur_value, c = color[2*j], alpha= 0.9) 
                plt.plot(self.m_episode, self.mean_value, c = color[2*j+1], label= label[j], alpha= 0.7)
                plt.fill_between(self.m_episode, self.lower_value, self.upper_value, color = color[2*j], alpha=0.4)
            else: 
                # plt.plot(self.m_episode, self.cur_value, c = color[2*j], alpha= 0.3) 
                plt.plot(self.m_episode, self.mean_value, c = color[2*j+1], label= label[j], alpha= 0.5)
                plt.fill_between(self.m_episode, self.lower_value, self.upper_value, color = color[2*j], alpha=0.2)
            print(label[j],self.name,round(max(self.mean_value),2))

            self.episode = []
            self.m_episode = []
            self.cur_value =  []
            self.mean_value =  []
            self.upper_value =  []
            self.lower_value =  []

        if name == 'Total':
            plt.legend(loc=2)
        plt.grid(True)
        fig_path = os.path.join(os.path.dirname(__file__), name+'_Reward_Compare.png')
        plt.savefig(fig_path)
        print("plot")
        

if __name__ == '__main__':
    print_value = ['Goalkeeper', 'Defenders', 'Forwards', 'Team', 'Total']
    for value in print_value:
        main = Plot(value)
        main.run(value)