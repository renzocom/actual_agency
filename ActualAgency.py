
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
import networkx as nx


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

def get_genome(genomes, run, agent):
    genome = genomes[run]['GENOME_root::_sites'][agent]
    genome = np.squeeze(np.array(np.matrix(genome)))
    return genome

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


def parseActivity(path,file,n_runs=30,n_agents=61,n_trials=64,world_height=35,n_nodes=8,n_sensors=2,n_hidden=4,n_motors=2):
    with open(os.path.join(path,file),'rb') as f:
        activity = pickle.load(f)

    all_activity = np.zeros((n_runs,n_agents,n_trials,world_height,n_nodes),dtype=int)
    for i in range(n_runs):
        print('{}/{}'.format(i+1,n_runs))
        all_activity[i,:,:,:,:] = getBrainActivity(activity[i],n_agents,n_trials,n_nodes,n_sensors,n_hidden,n_motors)

    with open(os.path.join(path,'activity_array.pkl'),'wb') as f:
        pickle.dump(all_activity, f)

    return all_activity


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

    
def AnalyzeTransitions(network, activity, cause_indices=[0,1,4,5,6,7], effect_indices=[2,3], 
                       sensor_indices=[0,1], motor_indices=[2,3],
                       purview = [],alpha = [],motorstate = [],transitions = []):
    
    states = len(activity)
    n_nodes = len(activity[0])
    x_indices = [i for i in range(n_nodes) if i not in motor_indices]
    y_indices = [i for i in range(n_nodes) if i not in sensor_indices]
    
    if len(transitions)>0:
        tran = [np.append(transitions[i][0][x_indices],transitions[i][1][y_indices]) for i in list(range(0,len(transitions)))]
    else:
        tran = []
        
    for s in list(range(states-1)):
        # 2 sensors
        x = activity[s,:].copy()
        x[motor_indices] = [0]*len(motor_indices)
        y = activity[s+1,:].copy()
        y[sensor_indices] = [0]*len(sensor_indices)
        
        occurence = np.append(x[x_indices],y[y_indices]).tolist()

        # checking if this transition has never happened before for this agent
        if not any([occurence == t.tolist() for t in tran]):
            # generating a transition
            transition = pyphi.actual.Transition(network, x, y, cause_indices, 
                effect_indices, cut=None, noise_background=False)
            CL = transition.find_causal_link(pyphi.Direction.CAUSE, tuple(effect_indices), purviews=False, allow_neg=False)

            alpha.append(CL.alpha)
            purview.append(CL.purview)
            motorstate.append(tuple(y[motor_indices]))

            # avoiding recalculating the same occurence twice
            tran.append(np.array(occurence))
            transitions.append([np.array(x),np.array(y)])
            
    return purview, alpha, motorstate, transitions

def createPandasFromACAnalysis(LODS,agents,activity,TPMs,CMs,labs):
    catch = list(range(0,31))
    avoid = list(range(32,64))

    purview_catch = []
    alpha_catch = []
    motor_catch = []
    transitions_catch = []

    purview_avoid = []
    alpha_avoid = []
    motor_avoid = []
    transitions_avoid = []

    for lod in LODS:
        purview_catch_LOD = []
        alpha_catch_LOD = []
        motor_catch_LOD = []
        transitions_catch_LOD = []

        purview_avoid_LOD = []
        alpha_avoid_LOD = []
        motor_avoid_LOD = []
        transitions_avoid_LOD = []

        for agent in agents: 
            print('LOD: {} out of {}'.format(lod,np.max(LODS)))
            print('agent: {} out of {}'.format(agent,np.max(agents)))
            purview_catch_agent = []
            alpha_catch_agent = []
            motor_catch_agent = []
            transitions_catch_agent = []

            purview_avoid_agent = []
            alpha_avoid_agent = []
            motor_avoid_agent = []
            transitions_avoid_agent = []

            tr = []
            TPM = np.squeeze(TPMs[lod,agent,:,:])
            CM = np.squeeze(CMs[lod,agent,:,:])
            TPMmd = pyphi.convert.to_multidimensional(TPM)
            network_2sensor = pyphi.Network(TPMmd, cm=CM, node_labels=labs)

            for t in catch:
                purview_catch_agent, alpha_catch_agent, motor_catch_agent, transitions_catch_agent = AnalyzeTransitions(
                    network_2sensor, np.squeeze(activity[lod,agent,t,:,:]),
                    purview = purview_catch_agent, alpha = alpha_catch_agent, 
                    motorstate = motor_catch_agent, transitions=transitions_catch_agent)
            for t in avoid:
                purview_avoid_agent, alpha_avoid_agent, motor_avoid_agent, transitions_avoid_agent = AnalyzeTransitions(
                    network_2sensor, np.squeeze(activity[lod,agent,t,:,:]),
                    purview = purview_avoid_agent, alpha = alpha_avoid_agent, 
                    motorstate = motor_avoid_agent, transitions=transitions_avoid_agent)

            purview_catch_LOD.append(purview_catch_agent)
            alpha_catch_LOD.append(alpha_catch_agent)
            motor_catch_LOD.append(motor_catch_agent)
            transitions_catch_LOD.append(transitions_catch_agent)

            purview_avoid_LOD.append(purview_avoid_agent)
            alpha_avoid_LOD.append(alpha_avoid_agent)
            motor_avoid_LOD.append(motor_avoid_agent)
            transitions_avoid_LOD.append(transitions_avoid_agent)  

        purview_catch.append(purview_catch_LOD)
        alpha_catch.append(alpha_catch_LOD)
        motor_catch.append(motor_catch_LOD)
        transitions_catch.append(transitions_catch_LOD)

        purview_avoid.append(purview_avoid_LOD)
        alpha_avoid.append(alpha_avoid_LOD)
        motor_avoid.append(motor_avoid_LOD)
        transitions_avoid.append(transitions_avoid_LOD)    

    purview = []
    alpha = []
    motor = []
    catch = []
    transitions = []
    lod_aux = []
    agent_aux = []
    s1 = []
    s2 = []
    h1 = []
    h2 = []
    h3 = []
    h4 = []

    idx = 0

    for lod in list(range(0,len(LODS))):
        for agent in list(range(0,len(agents))): 
            for i in list(range(len(purview_catch[lod][agent]))):
                
                motor.append(np.sum([ii*(2**idx) for ii,idx in zip(motor_catch[lod][agent][i],list(range(0,len(motor_catch[lod][agent][i]))))]))
                catch.append(1)
                transitions.append(transitions_catch[lod][agent][i])
                lod_aux.append(lod)
                agent_aux.append(agent)
                    
                if purview_catch[lod][agent][i] is not None:
                    purview.append([labs_2sensor[ii] for ii in purview_catch[lod][agent][i]])
                    s1.append(1 if 's1' in purview[idx] else 0)
                    s2.append(1 if 's2' in purview[idx] else 0)
                    h1.append(1 if 'h1' in purview[idx] else 0)
                    h2.append(1 if 'h2' in purview[idx] else 0)
                    h3.append(1 if 'h3' in purview[idx] else 0)
                    h4.append(1 if 'h4' in purview[idx] else 0)
                    alpha.append(alpha_catch[lod][agent][i])
                    idx+=1

                else:
                    purview.append('none') 
                    alpha.append(alpha_catch[lod][agent][i])
                    s1.append(0)
                    s2.append(0)
                    h1.append(0)
                    h2.append(0)
                    h3.append(0)
                    h4.append(0)
                    idx+=1

            for i in list(range(len(purview_avoid[lod][agent]))):
                
                motor.append(np.sum([ii*(2**idx) for ii,idx in zip(motor_avoid[lod][agent][i],list(range(0,len(motor_avoid[lod][agent][i]))))]))
                catch.append(0)
                transitions.append(transitions_avoid[lod][agent][i])
                lod_aux.append(lod)
                agent_aux.append(agent)
                    
                if purview_avoid[lod][agent][i] is not None:
                    purview.append([labs_2sensor[ii] for ii in purview_avoid[lod][agent][i]])
                    s1.append(1 if 's1' in purview[idx] else 0)
                    s2.append(1 if 's2' in purview[idx] else 0)
                    h1.append(1 if 'h1' in purview[idx] else 0)
                    h2.append(1 if 'h2' in purview[idx] else 0)
                    h3.append(1 if 'h3' in purview[idx] else 0)
                    h4.append(1 if 'h4' in purview[idx] else 0)
                    alpha.append(alpha_avoid[lod][agent][i])
                    idx+=1

                else:
                    purview.append('none') 
                    alpha.append(alpha_avoid[lod][agent][i])
                    s1.append(0)
                    s2.append(0)
                    h1.append(0)
                    h2.append(0)
                    h3.append(0)
                    h4.append(0)
                    idx+=1

    dictforpd = {'purview':purview,
                    'motor':motor,
                    'alpha':alpha,
                    's1':s1,
                    's2':s2,
                    'h1':h1,
                    'h2':h2,
                    'h3':h3,
                    'h4':h4,
                    'motor':motor,
                    'catch': catch,
                    'transition': transitions,
                    'LOD': lod_aux,
                    'agent': agent_aux,
                    }  

    panda = pd.DataFrame(dictforpd)
    
    return panda
    

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


def hist2d_2LODdata(LODdata1x,LODdata1y,LODdata2x,LODdata2y, nbins=20):
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

    xbins = np.linspace(xmin,xmax,nbins)
    ybins = np.linspace(ymin,ymax,nbins)
    plt.figure()
    plt.subplot(121)
    plt.hist2d(np.ravel(LODdata1x),np.ravel(LODdata1y),[xbins,ybins],norm=mpl.colors.LogNorm())
    plt.subplot(122)
    plt.hist2d(np.ravel(LODdata2x),np.ravel(LODdata2y),[xbins,ybins],norm=mpl.colors.LogNorm())


### OTHER FUNCTIONS
def plot_brain(cm, state=None, ax=None):
    n_nodes = cm.shape[0]
    if n_nodes==7:
        labels = ['S1','M1','M2','A','B','C','D']
        pos = {'S1': (5,40), #'S2': (20, 40),
           'A': (0, 30), 'B': (20, 30),
           'C': (0, 20), 'D': (20, 20),
          'M1': (5,10), 'M2': (15,10)}
        nodetype = (0,1,1,2,2,2,2)

        ini_hidden = 3

    elif n_nodes==8:
        labels = ['S1','S2','M1','M2','A','B','C','D']
        pos = {'S1': (5,40), 'S2': (15, 40),
           'A': (0, 30), 'B': (20, 30),
           'C': (0, 20), 'D': (20, 20),
          'M1': (5,10), 'M2': (15,10)}
        nodetype = (0,0,1,1,2,2,2,2)
        ini_hidden = 4

    state = [1]*n_nodes if state==None else state

    G = nx.from_numpy_matrix(cm, create_using=nx.DiGraph())

    mapping = {key:x for key,x in zip(range(n_nodes),labels)}
    G = nx.relabel_nodes(G, mapping)

    blue, red, green, grey, white = '#77b3f9', '#f98e81', '#8abf69', '#adadad', '#ffffff'
    blue_off, red_off, green_off, grey_off = '#e8f0ff','#ffe9e8', '#f0ffe8', '#f2f2f2'

    colors = np.array([red, blue, green, grey, white])
    colors = np.array([[red_off,blue_off,green_off, grey_off, white],
                       [red,blue,green, grey, white]])

    node_colors = [colors[state[i],nodetype[i]] for i in range(n_nodes)]
    # Grey Uneffective or unaffected nodes
    cm_temp = copy.copy(cm)
    cm_temp[range(n_nodes),range(n_nodes)]=0
    unaffected = np.where(np.sum(cm_temp,axis=0)==0)[0]
    uneffective = np.where(np.sum(cm_temp,axis=1)==0)[0]
    noeffect = list(set(unaffected).union(set(uneffective)))
    noeffect = [ix for ix in noeffect if ix in range(ini_hidden,ini_hidden+4)]
    node_colors = [node_colors[i] if i not in noeffect else colors[state[i],3] for i in range(len(G.nodes))]

    #   White isolate nodes
    isolates = [x for x in nx.isolates(G)]
    node_colors = [node_colors[i] if labels[i] not in isolates else colors[0,4] for i in range(len(G.nodes))]

    self_nodes = [labels[i] for i in range(n_nodes) if cm[i,i]==1]
    linewidths = [2.5 if labels[i] in self_nodes else 1 for i in range(n_nodes)]

#     fig, ax = plt.subplots(1,1, figsize=(4,6))
    nx.draw(G, with_labels=True, node_size=800, node_color=node_colors,
    edgecolors='#000000', linewidths=linewidths, pos=pos, ax=ax)
