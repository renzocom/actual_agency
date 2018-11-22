import numpy as np
import time
from pathlib import Path
import pandas as pd
import os
import copy

class Animat:
    def __init__(self, params):
        self.n_left_sensors = params['nrOfLeftSensors']
        self.n_right_sensors = params['nrOfRightSensors']
        self.n_sensors = self.n_right_sensors + self.n_left_sensors
        self.n_hidden = params['hiddenNodes']
        self.n_motors = 2
        self.n_nodes = self.n_left_sensors + self.n_right_sensors + self.n_hidden + self.n_motors
        self.gapwidth = params['gapWidth']
        self.length = self.n_left_sensors + self.n_right_sensors + self.gapwidth
        self.x = params['x'] if 'x' in params else 0
        self.y = params['y'] if 'y' in params else 0

    def __len__(self):
        return self.length

    def set_x(self, position):
        self.x = position
    def set_y(self, position):
        self.y = position

    def _getBrainActivity(self,data):
        world_height = 34
        print('Creating activity matrix from MABE otput...')
        n_trials = int((np.size(data,0))/world_height)
        brain_activity = np.zeros((n_trials,1+world_height,self.n_nodes))

        for i in list(range(n_trials)):
            for j in list(range(world_height+1)):
                ix = i*world_height + j
                if j==0:
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:self.n_sensors]
                    hidden = np.zeros(self.n_hidden)
                    motor = np.zeros(self.n_motors)
                elif j==world_height:
                    sensor = np.zeros(self.n_sensors)
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                else:
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:self.n_sensors]
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                nodes = np.r_[sensor, motor, hidden]
                brain_activity[i,j,:] = nodes
        print('Done.')
        return brain_activity

    def saveBrainActivity(self, brain_activity):
        if type(brain_activity)==pd.core.frame.DataFrame:
            self.brain_activity = self._getBrainActivity(brain_activity)
        else: ## array
            assert brain_activity.shape[2]==self.n_nodes, "Brain history does not match number of nodes = {}".format(self.n_nodes)
            self.brain_activity = np.array(brain_activity)

    def getMotorActivity(self, trial):
        motor_states = self.brain_activity[trial,:,self.n_sensors:self.n_sensors+2]
        motor_activity = []
        for state in motor_states:
            state = list(state)
            if state==[0,0] or state==[1,1]:
                motor_activity.append(0)
            elif state==[1,0]:
                motor_activity.append(1)
            else: # state==[0,1]
                motor_activity.append(-1)
        return motor_activity


class Block:
    def __init__(self, size, direction, block_type, ini_x, ini_y=0):
        self.size = size
        self.direction = direction
        self.type = block_type
        self.x = ini_x
        self.y = ini_y

    def __len__(self):
        return self.size

    def set_x(self, position):
        self.x = position
    def set_y(self, position):
        self.y = position

class Screen:
    def __init__(self, width, height):
        self.screen = np.zeros((height + 1,width))
        self.width = width
        self.height = height
        self.screen_history = np.array([])

    def resetScreen(self):
        self.screen = np.zeros(self.screen.shape)
        self.screen_history = np.array([])

    def drawAnimat(self, animat):
        self.screen[-1,:] = 0
        self.screen[-1,self.wrapper(range(animat.x,animat.x+len(animat)))] = 1

    def drawBlock(self, block):
        self.screen[:-1,:] = 0
        self.screen[block.y,self.wrapper(range(block.x, block.x+len(block)))] = 1

    def saveCurrentScreen(self):
        if len(self.screen_history)==0:
            self.screen_history = copy.copy(self.screen[np.newaxis,:,:])
        else:
            self.screen_history = np.r_[self.screen_history, self.screen[np.newaxis,:,:]]

    def wrapper(self,index):
        if not hasattr(index, '__len__'):
            return index%self.width
        else:
            return [ix%self.width for ix in index]

class World:
    def __init__(self, width=16, height=35):
        self.width = width # ComplexiPhi world is 35 (and not 34! 34 is the number of updates)
        self.height = height
        self.screen = Screen(self.width, self.height)

    def _runGameTrial(self, trial, animat, block):

        total_time = self.height # 35 time steps, 34 updates
        motor_activity = animat.getMotorActivity(trial)

        # t=0 # Initial position (game hasn't started yet.)
        self.screen.resetScreen()
        self.screen.drawAnimat(animat)
        self.screen.drawBlock(block)
        self.screen.saveCurrentScreen()

        for t in range(1, total_time):

            animat.x = self.screen.wrapper(animat.x + motor_activity[t])

            if t<total_time:
                if block.direction == 'right':
                    block.x = self.screen.wrapper(block.x + 1)
                else:
                    block.x = self.screen.wrapper(block.x - 1)

                block.y = block.y + 1

            self.screen.drawAnimat(animat)
            self.screen.drawBlock(block)
            self.screen.saveCurrentScreen()
        # animat catches the block if it overlaps with it in t=34
        win = self._check_win(block, animat)

        return self.screen.screen_history, win

    def _getInitialCond(self, trial):
        animal_init_x = trial % self.width
        self.animat.set_x(animal_init_x)

        block_size = self.block_types[trial //(self.width * 2)]
        block_direction = 'left' if (trial // self.width) % 2 == 0 else 'right'
        block_value = 'catch' if (trial // (self.width * 2)) % 2 == 0 else 'avoid'
        block = Block(block_size, block_direction, block_value, 0)

        return self.animat, block

    def runFullGame(self, block_types, brain_history, animat_params):
        self.block_types = block_types
        self.n_trials = self.width * 2 * len(block_types)

        self.animat = Animat(animat_params)
        self.animat.saveBrainActivity(brain_history)

        self.history = np.zeros((self.n_trials,self.height,self.height+1,self.width))

        wins = []
        for trial in range(self.n_trials):
            self.animat, block = self._getInitialCond(trial)
            self.history[trial,:,:,:], win = self._runGameTrial(trial,self.animat, block)
            wins.append(win)
        return self.history, wins

    def _check_win(self, block, animat):
        block_ixs = self.screen.wrapper(range(block.x, block.x + len(block)))
        animat_ixs = self.screen.wrapper(range(animat.x, animat.x + len(animat)))
        catch = True if len(set(block_ixs).intersection(animat_ixs))>0 else False
        win = True if (block.type=='catch' and catch) or (block.type=='avoid' and not catch) else False
        return win

    def getFinalScore(self):
        score = 0
        for trial in range(self.n_trials):
            animat, block = self._getInitialCond(trial)
            # print('trial {}'.format(trial))
            # print('A0: {} B0: {} ({}, {}, {})'.format(animat.x,block.x,len(block),block.direction, block.type))

            animat.x = self.screen.wrapper(animat.x + np.sum(animat.getMotorActivity(trial)[:]))

            direction = -1 if block.direction=='left' else 1
            block.x = self.screen.wrapper(block.x + (self.height-1)*direction)

            win = 'WIN' if self._check_win(block, animat) else 'LOST'
            # print('Af: {} Bf: {}'.format(animat.x, block.x))
            # print(win)
            # print()
            score += int(self._check_win(block, animat))
        print('Score: {}/{}'.format(score, self.n_trials))
        return score
