#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import random
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Participant, Game, Frame
except ImportError as err:
    print('player_random-walk: \'participant\' module cannot be imported:', err)
    raise

import pickle
import torch
import time
import math
import numpy as np

import helper
from drqn import DRQN
from rl_utils import  Logger, get_action, get_reward, get_team_reward, get_state, get_global_state, TouchCounter



#reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5
GOALKICK = 6
CORNERKICK = 7
PENALTYKICK = 8
HALFTIME = 9
EPISODE_END = 10

#game_state
STATE_DEFAULT = 0
STATE_KICKOFF = 1
STATE_GOALKICK = 2
STATE_CORNERKICK = 3
STATE_PENALTYKICK = 4

#coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
Z = 2
TH = 3
ACTIVE = 4
TOUCH = 5
BALL_POSSESSION = 6
robot_size = 0.15

#rodot_index
GK_INDEX = 0 
D1_INDEX = 1 
D2_INDEX = 2 
F1_INDEX = 3 
F2_INDEX = 4

CHECKPOINT_GK = os.path.join(os.path.dirname(__file__), 'models/Robot_GK.th')
CHECKPOINT_D12 = os.path.join(os.path.dirname(__file__), 'models/Robot_D12.th')
CHECKPOINT_F12 = os.path.join(os.path.dirname(__file__), 'models/Robot_F12.th')
CHECKPOINT = [CHECKPOINT_GK, CHECKPOINT_D12, CHECKPOINT_F12]

CHECKPOINT_MIXER_D12 = os.path.join(os.path.dirname(__file__), 'models/mixer_D12.th')
CHECKPOINT_MIXER_F12 = os.path.join(os.path.dirname(__file__), 'models/mixer_F12.th')
CHECKPOINT_MIXER = [CHECKPOINT_MIXER_D12, CHECKPOINT_MIXER_F12]

config_path = os.path.join(os.path.dirname(__file__), 'config.pickle')
memory_path = os.path.join(os.path.dirname(__file__), 'memory.pickle')

class Frame(object):
    def __init__(self):
        self.time = None
        self.score = None
        self.reset_reason = None
        self.game_state = None
        self.subimages = None
        self.coordinates = None
        self.half_passed = None

class TestPlayer(Participant):
    def init(self, info):
        self.field = info['field']
        self.max_linear_velocity = info['max_linear_velocity']
        self.goal = info['goal']
        self.penalty_area = info['penalty_area']
        self.goal_area = info['goal_area']
        self.number_of_robots = info['number_of_robots']
        self.end_of_frame = False
        self._frame = 0 
        self.wheels = [ 0 for _ in range(30)]
        self.cur_posture = []
        self.prev_posture = []
        self.cur_posture_opp = []
        self.cur_ball = []
        self.prev_ball = []

        self.previous_frame = Frame()
        self.frame_skip = 2 # number of frames to skip
        self.epi_max_len = 40
        self.obs_size = 30 #243 #37 for usual state #243 for lidar state
        self.state_size = 22
        self.act_size = 20
        self.role_type = 3
        self.mixer_num = 2
        self.episode_observation = [[[0 for _ in range(self.obs_size)] for _ in range(self.number_of_robots)] for _ in range(self.epi_max_len)]
        self.episode_state = [[0 for _ in range(self.state_size)] for _ in range(self.epi_max_len)]
        self.episode_action = [[0 for _ in range(self.number_of_robots)] for _ in range(self.epi_max_len)]
        self.episode_reward = [[0 for _ in range(self.role_type)] for _ in range(self.epi_max_len)]
        self.episode_team_reward = [0 for _ in range(self.epi_max_len)]
        self.episode_mask = [0 for _ in range(self.epi_max_len)]
        self.t = 0
        # for RL
        self.action = [0 for _ in range(self.number_of_robots)]
        self.pre_action = [0 for _ in range(self.number_of_robots)]

        self.num_inputs = self.obs_size
        self.load = False
        self.epsilon = 0.95
        self.episode = 0
        with open(config_path,"rb") as fr:
            config_data = pickle.load(fr)
            self.load = config_data[0]
            self.epsilon = config_data[1]
    
        self.trainer = DRQN(self.number_of_robots, self.obs_size, self.state_size, self.act_size, self.epi_max_len, self.epsilon, self.load)
        self.trainer.init_hidden()
        self.init_hidden_states = [[] for _ in range(self.role_type)]
        self.init_hidden_states = None

        self.total_reward = 0
        self.reward = [0 for _ in range(self.number_of_robots)]
        self.team_reward = 0
        self.rew =np.zeros((self.number_of_robots,4))
        self.touch_counter = TouchCounter(self.number_of_robots)
        self.printConsole("Initializing variables...")

    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_posture = received_frame.coordinates[MY_TEAM]
        self.cur_posture_opp = received_frame.coordinates[OP_TEAM]
        self.prev_posture = self.previous_frame.coordinates[MY_TEAM]
        self.prev_posture_opp = self.previous_frame.coordinates[OP_TEAM]
        self.prev_ball = self.previous_frame.coordinates[BALL]

    def init_episode(self):
        self.episode_observation = [[[0 for _ in range(self.obs_size)] for _ in range(self.number_of_robots)] for _ in range(self.epi_max_len)]
        self.episode_state = [[0 for _ in range(self.state_size)] for _ in range(self.epi_max_len)]
        self.episode_action = [[0 for _ in range(self.number_of_robots)] for _ in range(self.epi_max_len)]
        self.episode_reward = [[0 for _ in range(self.role_type)] for _ in range(self.epi_max_len)]
        self.episode_team_reward = [0 for _ in range(self.epi_max_len)]
        self.episode_mask = [0 for _ in range(self.epi_max_len)]
        self.t = 0

    def save_experience(self, observation, state, init_hidden, action, reward, team_reward, mask):
        with open(memory_path,"wb") as fw:
            pickle.dump([observation,state,init_hidden,action,reward,team_reward,mask],fw)

    def update(self, received_frame):

        if received_frame.end_of_frame:
        
            self._frame += 1

            if (self._frame == 1):
                self.previous_frame = received_frame
                self.get_coord(received_frame)

            self.get_coord(received_frame)
            self.touch_counter.Counts(self.cur_posture, received_frame.reset_reason)

            ## reward ##
            rew_n = [ get_reward(self.cur_posture, self.prev_posture, self.cur_ball, self.prev_ball, self.field, self.goal, i, self.touch_counter, received_frame.reset_reason)  for i in range(self.number_of_robots) ]
            for i in range(self.number_of_robots):
                self.rew[i][self._frame % self.frame_skip] = rew_n[i]
            self.rew_n = [ (self.rew[i][0] + self.rew[i][1] + self.rew[i][2] + self.rew[i][3] ) / self.frame_skip for i in range(self.number_of_robots) ]
            self.reward = [round(self.rew_n[i], 3) for i in range(self.number_of_robots)]
            self.team_reward = round(get_team_reward(self.cur_posture, self.prev_posture, self.cur_ball, self.prev_ball, self.field, self.goal), 3)

            ## episode ##
            if self._frame % self.frame_skip == 1:
                state = get_state(self.cur_posture, self.prev_posture, self.cur_posture_opp, self.prev_posture_opp, self.cur_ball, self.prev_ball, self.field, self.goal, self.max_linear_velocity) # when use state
                global_state = get_global_state(self.cur_posture, self.prev_posture, self.cur_posture_opp, self.prev_posture_opp, self.cur_ball, self.prev_ball, self.field, self.goal, self.max_linear_velocity) # when use state

                for i in range(self.number_of_robots):
                    self.episode_observation[self.t][i] = state[i] # when use state

                self.episode_observation = np.reshape([self.episode_observation],(self.epi_max_len, self.number_of_robots, self.obs_size))
                
                self.episode_state[self.t] = global_state
                self.episode_state = np.reshape([self.episode_state],(self.epi_max_len, self.state_size))

                act_input = self.episode_observation[self.t]
                self.action = self.trainer.select_action(act_input)

                for i in range(self.number_of_robots):
                    self.episode_action[self.t][i] = self.action[i]    
                self.episode_action = np.reshape([self.episode_action],(self.epi_max_len, self.number_of_robots, 1))
                
                self.episode_reward[self.t][0] = self.reward[GK_INDEX]
                self.episode_reward[self.t][1] = self.reward[D1_INDEX] + self.reward[D2_INDEX]
                self.episode_reward[self.t][2] = self.reward[F1_INDEX] + self.reward[F2_INDEX]
                self.episode_reward = np.reshape([self.episode_reward],(self.epi_max_len, self.role_type, 1))


                self.episode_team_reward[self.t] = self.team_reward
                self.episode_team_reward = np.reshape([self.episode_team_reward],(self.epi_max_len, 1))
 

                self.episode_mask[self.t] = 1 if received_frame.reset_reason == NONE else 0
                self.episode_mask = np.reshape([self.episode_mask],(self.epi_max_len, 1))
                if self.t == 0:
                    self.init_hidden_GK = self.trainer.hidden_states[0].cpu().data.numpy().tolist() 
                    self.init_hidden_D12 = self.trainer.hidden_states[1].cpu().data.numpy().tolist() 
                    self.init_hidden_F12 = self.trainer.hidden_states[2].cpu().data.numpy().tolist() 
                    self.init_hidden_states = self.init_hidden_GK + self.init_hidden_D12 + self.init_hidden_F12
                    self.init_hidden_states = np.reshape([self.init_hidden_states],(self.number_of_robots, -1))
               
                self.total_reward += sum(self.reward) + self.team_reward

                self.t +=1

            else:
                self.action = self.pre_action

            ## set speed ##
            for role in range(self.number_of_robots):
                self.wheels[6*role:6*role+6] = get_action(role, self.action[role], self.max_linear_velocity)

            self.set_speeds(self.wheels)

            ## episode end ##
            if (received_frame.reset_reason > 1):
                # self.printConsole("init_hidden")
                self.trainer.init_hidden()
            if (((received_frame.reset_reason > 1) and (self.t >= int(self.epi_max_len/5))) or ((self.t % self.epi_max_len == 0) and (self.t != 0))):
   
                self.save_experience(self.episode_observation, self.episode_state, self.init_hidden_states, self.episode_action, self.episode_reward, self.episode_team_reward, self.episode_mask)
                self.init_episode()
                self.total_reward = 0
                self.episode += 1
                try:
                    if self.episode%20 == 0:
                        ## load config data ##
                        with open(config_path,"rb") as fr:
                            config_data = pickle.load(fr)
                            self.load = config_data[0]
                            self.epsilon = config_data[1]
                        self.trainer.epsilon = self.epsilon
                    ## load agents network ##
                    for role in range(self.role_type):
                        self.trainer.net[role].load_state_dict(torch.load(CHECKPOINT[role]))
                except:    
                    self.printConsole("loading err occur")
                    pass

            elif ((received_frame.reset_reason > 1) and (self.t < int(self.epi_max_len/5))):
                self.init_episode()
                self.total_reward = 0

            self.end_of_frame = False
            self.pre_action = self.action
            self.previous_frame = received_frame

if __name__ == '__main__':
    player = TestPlayer()
    player.run()