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

data_path1 = os.path.join(os.path.dirname(__file__), 'SCORE_TSHCT.pickle')
data_path11 = os.path.join(os.path.dirname(__file__), 'SCORE_TSHCT_Dueling.pickle')
data_path2 = os.path.join(os.path.dirname(__file__), 'SCORE_QMIX.pickle')
data_path3 = os.path.join(os.path.dirname(__file__), 'SCORE_VDN.pickle')
data_path4 = os.path.join(os.path.dirname(__file__), 'SCORE_COMA.pickle')
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
        with open(data_path11,"rb") as fr:
            self.value11 = pickle.load(fr)
        with open(data_path2,"rb") as fr:
            self.value2 = pickle.load(fr)
        with open(data_path3,"rb") as fr:
            self.value3 = pickle.load(fr)
        with open(data_path4,"rb") as fr:
            self.value4 = pickle.load(fr)
        self.cur_num = 1
        self.num = 10
        self.value = [self.value1,self.value11,self.value2,self.value3,self.value4]


        print("Initializing variables...")

    def run(self,name,fig_path):
        plt.clf()
        for r in [0,1,4,3,2]:
        # for r in range(5):
            score =[]
            scored = []
            zip_object = zip(self.value[r][0], self.value[r][1])
            for score_i, scored_i in zip_object:
                score.append(score_i)
                scored.append(-scored_i)
            num_point = len(score)//5
            score_1,scored_1 = sum(score[:num_point]),sum(scored[:num_point]) 
            score_2,scored_2 = sum(score[num_point:2*num_point]),sum(scored[num_point:2*num_point]) 
            score_3,scored_3 = sum(score[2*num_point:3*num_point]),sum(scored[2*num_point:3*num_point]) 
            score_4,scored_4 = sum(score[3*num_point:4*num_point]),sum(scored[3*num_point:4*num_point]) 
            score_5,scored_5 = sum(score[4*num_point:]),sum(scored[4*num_point:]) 
            num_score = [score_1,score_2,score_3,score_4,score_5] 
            num_scored = [scored_1,scored_2,scored_3,scored_4,scored_5]
            sum_score=[]
            sum_scored=[]
            ratio = []
            sum_num = 20
            score_and_scored =0
            for s in range(len(score)):
                if s< sum_num:
                    sum_score.append(sum(score[:s]))
                    sum_scored.append(sum(scored[:s]))
                    score_and_scored = sum(score[:s]) - sum(scored[:s])
                    if score_and_scored ==0:
                        ratio.append(0)
                    else:
                        ratio.append(100*sum(score[:s])/score_and_scored)
                else:
                    sum_score.append(sum(score[s-sum_num:s]))
                    sum_scored.append(sum(scored[s-sum_num:s]))
                    score_and_scored = sum(score[s-sum_num:s]) - sum(scored[s-sum_num:s])
                    if score_and_scored ==0:
                        ratio.append(0)
                    else:
                        ratio.append(100*sum(score[s-sum_num:s])/score_and_scored)
            for i in range(len(ratio)):
                self.episode.append(i+1)
                if i >= self.num:
                    self.m_episode.append(self.episode[i]-5)
                    self.mean_value.append( np.mean(ratio[i-self.num:i]))
                    self.upper_value.append(  max(ratio[i-self.num:i]))
                    self.lower_value.append(  min(ratio[i-self.num:i]))
            xlab = 'Episodes'
            ylab1 = 'Score and Concede'
            ylab2 = 'Score-Concede Rate'
            color=['pink','red','thistle','indigo','lightskyblue','b','lightgreen','green','moccasin','orange']
            if r == 0:
                fig = plt.figure(figsize=(20, 10))
                ax1 = fig.add_subplot(2,1,1)
                ax2 = fig.add_subplot(2,1,2)
                ax1.set_title(name,fontsize=30)
                # ax1.set_xlabel(xlab,fontsize=20)
                ax2.set_xlabel(xlab,fontsize=25)
                ax1.set_ylabel(ylab1,fontsize=22)
                ax2.set_ylabel(ylab2,fontsize=22)
                ax1.grid(True)
                ax2.grid(True)
                ax1.set_xticks([2, 9, 16, 23, 30])
                ax1.set_xticklabels(['0k ~ 40k', '40k ~ 80k', '80k ~ 120k', '120k ~ 160k', '160k ~ 200k'], fontsize=20)
                ax1.set_yticks([-60, -40, -20, 0, 20, 40, 60, 80, 100])
                ax1.set_yticklabels(['60 Concede', '40 Concede', '20 Concede', '0', '20 Score', '40 Score', '60 Score', '80 Score', '100 Score' ], fontsize=15)
                ax2.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                ax2.set_xticklabels(['20k', '40k', '60k', '80k', '100k', '120k', '140k', '160k', '180k', '200k'], fontsize=20)
                ax2.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                ax2.set_yticklabels(['0', '10 %', '20 %', '30 %', '40 %', '50 %', '60 %', '70 %', '80 %', '90 %', '100 %' ], fontsize=15)
                label=['TSHCT-DRQN','TSHCT-Dueling DRQN','QMIX','VDN','COMA']
            loc = [0,1,4,3,2]
            ax1.bar(range(loc[r],35,7), num_score, color=color[2*r+1],label=label[r]+' Score', width = 0.6)
            ax1.bar(range(loc[r],35,7), num_scored, bottom=0, color =color[2*r], label=label[r]+' Concede', width = 0.6)
            ax2.plot(self.m_episode, self.mean_value, color = color[2*r+1], label= label[r]) 
            # if r==0: 
            #     ax2.fill_between(self.m_episode, self.lower_value, self.upper_value, color = color[2*r], alpha=1)
            # else:
            #     ax2.fill_between(self.m_episode, self.lower_value, self.upper_value, color = color[2*r], alpha=0.2)
            ax2.fill_between(self.m_episode, self.lower_value, self.upper_value, color = color[2*r], alpha=0.4)
            # ax1.legend(loc=2,fontsize=15)
            # ax2.legend(loc=2,fontsize=15)
            print(label[r],str('Sum-Score: ')+str(num_score[-1]),str('Sum-Concede: ')+str(-1*num_scored[-1]),str('Score Difference: ')+str(num_score[-1]+num_scored[-1]),str('Score Rate: ')+str(round(max(self.mean_value),2)))


            plt.savefig(fig_path)
            self.episode = []
            self.m_episode = []
            self.cur_value =  []
            self.mean_value =  []
            self.upper_value =  []
            self.lower_value =  []
            print("plot")
        

if __name__ == '__main__':
    print_value = ['Performance versus Evaluation Team 3' ]
    for value in print_value:
        fig_path = os.path.join(os.path.dirname(__file__), value+'_dueling.png')
        main = Plot(value)
        main.run(value,fig_path)