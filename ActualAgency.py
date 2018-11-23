import numpy as np
import pandas as pd
import subprocess as sp
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import numpy.random as ran
import copy
import pyphi
from pathlib import Path
import scipy.io as sio


### MABE RELATED FUNCTIONS
def parseTPM(TPM_jory):
    '''
    Function for parsing the output from the mabe TPMworld into a readable format
        Inputs:
            TPM_jory: (unpickled) csv output from mabe TPM-world 
        Outputs:
            allgates: A list of lists (num_agents long) containing all gates in the agents genome
    '''
    start = '{'
    end = '}'
    split = r'\r\n'
    s = str(TPM_jory)
    a=True

    s_byanimats = s.split(split)[1:-1]
    allgates = []
    for animat in s_byanimats:
        gates = []
        a = True
        while a:
            idx1 = str(animat).find(start)
            idx2 = str(animat).find(end)+1
            if idx1==-1:
                a = False
            else:
                gate_string = animat[idx1:idx2]
                gates.append(eval(gate_string))
                animat = animat[idx2+1:]
        allgates.append(gates)
    return allgates
    

def getBrainActivity(data, n_agents=1, n_trials=64, n_nodes=8, n_sensors=2,n_hidden=4,n_motors=2):
    '''
    Function for generating a activity matrices for the animats given outputs from mabe
        Inputs:
            data: a pandas object containing the mabe output from activity recording
            n_agents: number of agents recorded   
            n_trials: number of trials for each agent
            n_nodes: total number of nodes in the agent brain (sensors+motrs+hidden)
            n_sensors: number of sensors in the agent brain
            n_hidden: number of hidden nodes between the sensors and motors
            n_motors: number of motors in the agent brain
        Outputs:
            brain_activity: a matrix with the timeseries of activity for each trial of every agent. Dimensions(agents)
    '''
    print('Creating activity matrix from MABE otput...')
    world_height = 34
    brain_activity = np.zeros((n_agents,n_trials,1+world_height,n_nodes))

    for a in list(range(n_agents)):
        for i in list(range(n_trials)):
            for j in list(range(world_height+1)):
                ix = a*n_trials*world_height + i*world_height + j
                if j==0:
                    sensor = np.fromstring(data['input_LIST'][ix], dtype=int, sep=',')
                    hidden = np.zeros(n_hidden)
                    motor = np.zeros(n_motors)
                elif j==world_height:
                    sensor = np.zeros(n_sensors)
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                else:
                    sensor = np.fromstring(data['input_LIST'][ix], dtype=int, sep=',')
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                nodes = np.r_[sensor, motor, hidden]
                brain_activity[a,i,j,:] = nodes
    return brain_activity

### ACTTUAL CAUSATION ANALYSIS FUNCTIONS

def get_occurrences(activityData,numSensors,numHidden,numMotors):
    '''
    Function for converting activity data from mabe to past and current occurences.
        Inputs:
            activityData: array containing all activity data to be converted ((agent x) trials x time x nodes)
            numSensors: number of sensors in the agent brain
            numHidden: number of hiden nodes in the agent brain
            numMotors: number of motor units in the agent brain
        Outputs:
            x: past occurences (motor activity set to 0, since they have no effect on the future)
            y: current occurences (sensor activity set to 0, since they are only affected by external world)
    '''
    size = activityData.shape
    x = np.zeros(size)
    y = np.zeros(size)
    

    if len(size)==4:
        # deleting one timestep from each trial
        x = np.delete(x,(-1),axis=2)  
        y = np.delete(y,(-1),axis=2)
        
        # filling matrices with values
        x = copy.deepcopy(activityData[:,:,:-1,:])
        y = copy.deepcopy(activityData[:,:,1:,:])
        
        # setting sensors to 0 in y, and motors to zeros in x
        x[:,:,:,numSensors:numSensors+numMotors] = np.zeros(x[:,:,:,numSensors:numSensors+numMotors].shape)
        y[:,:,:,:numSensors] = np.zeros(y[:,:,:,:numSensors].shape)
    
    elif len(size)==3:
        # deleting one timestep from each trial
        x = np.delete(x,(-1),axis=1)  
        y = np.delete(y,(-1),axis=1) 
        
        # filling matrices with values
        x = copy.deepcopy(activityData[:,:-1,:])
        y = copy.deepcopy(activityData[:,1:,:])
        
        # setting sensors to 0 in y, and motors to zeros in x
        x[:,:,numSensors:numSensors+numMotors] = np.zeros(x[:,:,numSensors:numSensors+numMotors].shape)
        y[:,:,:numSensors] = np.zeros(y[:,:,:numSensors].shape) 
    
    return x, y


### DATA ANALYSIS FUNCTIONS

def Bootstrap_mean(data,n):
    '''
    Function for doing bootstrap resampling of the mean for a 2D data matrix.
        Inputs: 
            data: raw data samples to be bootsrap resampled (samples x datapoints)
            n: number of bootstrap samples to draw
        Outputs: 
            means: matrix containing all bootstrap samples of the mean (n x datapoints)
    '''    
    datapoints = len(data)
    timesteps = len(data[0])

    idx = list(range(0,n))
    means = [0 for i in idx]
    for i in idx:
        # drawing random timeseries (with replacement) from data
        bootstrapdata = np.array([data[d][:] for d in ran.choice(list(range(0,datapoints)),datapoints,replace=True)])
        means[i] = np.mean(bootstrapdata,0)
    
    return means
    
    
### PLOTTING FUNCTIONS

def plot_LODdata_and_Bootstrap(x,LODdata):
    '''
    Function for doing bootstrap resampling of the mean for a 2D data matrix.
        Inputs: 
            x:
            LODdata:
        Outputs: 
            fig:
    '''    

    fit = Bootstrap_mean(LODdata,500)
    m_fit = np.mean(fit,0)
    s_fit = np.std(fit,0) 
    fig = plt.figure(figsize=[20,10])
    for LOD in LODdata:
        plt.plot(x,LOD,'r',alpha=0.2)
    plt.plot(x,m_fit,'b')
    plt.plot(x,m_fit+s_fit,'b:')
    plt.plot(x,m_fit-s_fit,'b:')
    
    return fig
    
    
    
def plot_2LODdata_and_Bootstrap(x,LODdata1,LODdata2):
    '''
    Function for doing bootstrap resampling of the mean for a 2D data matrix.
        Inputs: 
            x:
            LODdata:
        Outputs: 
            fig:
    '''    

    fit1 = Bootstrap_mean(LODdata1,500)
    m_fit1 = np.mean(fit1,0)
    s_fit1 = np.std(fit1,0) 
    fit2 = Bootstrap_mean(LODdata2,500)
    m_fit2 = np.mean(fit2,0)
    s_fit2 = np.std(fit2,0) 
    fig = plt.figure(figsize=[20,10])
    for LOD1,LOD2 in zip(LODdata1,LODdata2):
        plt.plot(x,LOD1,'k',alpha=0.1)
        plt.plot(x,LOD2,'y',alpha=0.1)
    plt.plot(x,m_fit1,'k',linewidth=3)
    plt.plot(x,m_fit1+s_fit1,'k:',linewidth=2)
    plt.plot(x,m_fit1-s_fit1,'k:',linewidth=2)
    plt.plot(x,m_fit2,'y',linewidth=3)
    plt.plot(x,m_fit2+s_fit2,'y:',linewidth=2)
    plt.plot(x,m_fit2-s_fit2,'y:',linewidth=2)
    
    return fig


def hist2d_2LODdata(x,LODdata1x,LODdata1y,LODdata2x,LODdata2y):
    '''
    Function for doing bootstrap resampling of the mean for a 2D data matrix.
        Inputs: 
            x:
            LODdata:
        Outputs: 
            fig:
    '''
    xmin = np.min((np.min(LODdata1x),np.min(LODdata2x)))
    xmax = np.max((np.max(LODdata1x),np.max(LODdata2x)))
    ymin = np.min((np.min(LODdata1y),np.min(LODdata2y)))
    ymax = np.max((np.max(LODdata1y),np.max(LODdata2y)))
    
    xbins = np.linspace(xmin,xmax,20)
    ybins = np.linspace(ymin,ymax,20)
    plt.figure()
    plt.subplot(121)
    plt.hist2d(np.ravel(LODdata1x),np.ravel(LODdata1y),[xbins,ybins],norm=mpl.colors.LogNorm())
    plt.subplot(122)
    plt.hist2d(np.ravel(LODdata2x),np.ravel(LODdata2y),[xbins,ybins],norm=mpl.colors.LogNorm())
    
    
### OTHER FUNCTIONS

