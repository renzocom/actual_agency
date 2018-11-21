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


def get_states(n_nodes):
    '''
    Function for generating arrays with all possible states according to holi and loli indexing conventions.
        Inputs:
            n_nodes: number of nodes to calculate the full state state matrix for
        Outputs:
            states_holi: state by node (2**n x n) array containing all states in holi (big-endian) convention
            states_loli: state by node (2**n x n) array containing all states in loli (little-endian) convention
    '''
    states_holi = np.array(([list(('{:0'+str(n_nodes)+'d}').format(int(bin(x)[2:]))) for x in range(2**n_nodes)])).astype(int)
    states_loli = np.flip(states_holi,1)
    return states_holi, states_loli

def reduce_degenerate_outputs(gate_TPM, outputs):
    '''
    Reduces gate_TPM with degenerate outputs (e.g. outputs=[2,12,3,12] to outputs=[2,3,12]) by combining
    them with OR logic
        Inputs:
            gate_TPM: Original gate TPM (states x nodes) to be reduced
            outputs: IDs for the outputs the gate connects to (1 x nodes)
        Outputs:
            reduced_gate_TPM: Reduced gate TPM (states x nodes) now without degenerate outputs
            unique_outputs: IDs for the unique nodes the gate connects to (1 x nodes)
    '''
    # Find degenerate outputs
    unique_outputs = np.unique(outputs)
    unique_ixs = []

    if len(outputs)==len(unique_outputs):
        return gate_TPM, outputs

    for e in unique_outputs:
        ixs = list(np.where(outputs==e)[0])
        unique_ixs.append(ixs)

    # Reduce
    reduced_gate_TPM = np.zeros((gate_TPM.shape[0],len(unique_outputs)))
    for i in range(len(unique_outputs)):
        reduced_gate_TPM[:,i] = 1 - np.prod(1 - gate_TPM[:,unique_ixs[i]],1) # OR logic

    return reduced_gate_TPM, unique_outputs

def reduce_degenerate_inputs(gate_TPM, inputs, states_convention):
    '''
    Function for reducing gate_TPM with degenerate inputs (e.g. inputs=[2,12,3,12] to inputs=[2,3,12]) by removing
    input states that are internally inconsistent.
        Inputs:
            gate_TPM: the original gateTPM (states x nodes)
            inputs: IDs of inputs to the gate (1 x nodes)
            states_convention: specification of the covention used for state organizatino (loli or holi)
        Outputs:
            reduced_gate_TPM: the reduced gateTPM (states x nodes), now without degenerate inputs
            unique_inputs: IDs of unique inputs to the gate (1 x nodes)
            '''
    # Find degenerate outputs
    unique_inputs = np.unique(inputs)
    unique_ixs = []

    if len(unique_inputs)==len(inputs):
        return gate_TPM, inputs

    delete_row = []
    for e in unique_inputs:
        ixs = list(np.where(inputs==e)[0])
        unique_ixs.append(ixs)

    # making reduced holi
    input_holi, input_loli = get_states(len(inputs))
    input_states = input_holi if states_convention == 'holi' else input_loli

    # finding states where same node gives different values
    for ixs in unique_ixs:
        # check for duplicates
        if len(ixs)>1:
            # run through all states
            for i in list(range(0,len(input_states))):
                state = input_states[i,ixs]
                # check if activity of all duplicates match
                if not ((np.sum(state) == len(state)) or (np.sum(state) == 0)):
                    # remove row when they do not match
                    delete_row.append(i)
    reduced_gate_TPM = np.delete(gate_TPM,(delete_row),axis=0)
    return reduced_gate_TPM, unique_inputs


def expand_gate_TPM(gate_TPM, inputs, outputs, n_nodes, states_convention):
    '''
    Function for expanding the gate TPM (2**n_inputs x n_outputs) to the full size TPM (2**n_nodes x n_nodes).
        Inputs:
            gate_TPM: Original gate TPM to be expanded (states x nodes)
            inputs: IDs of inputs (1 x nodes)
            outputs: IDs of outputs (1 x nodes)
            n_nodes: total number of nodes in the agent
            states_convention: specification of convention used for state organization (holi or loli)
        Outputs:
            expanded_gate_TPM: Final gate TPM expanded to the size of the full agent TPM (state x node)
    '''
    full_holi, full_loli = get_states(n_nodes)
    full_states = full_holi if states_convention == 'holi' else full_loli

    n_inputs = len(inputs)
    gate_holi, gate_loli = get_states(n_inputs)
    gate_states = gate_holi if states_convention == 'holi' else gate_loli

    expanded_gate_TPM = np.zeros((2**n_nodes,n_nodes))

    for i in range(expanded_gate_TPM.shape[0]): # iterate through rows (inputs states)
        for j in range(gate_TPM.shape[0]):
            if np.all(gate_states[j,:]==full_states[i,inputs]):
                expanded_gate_TPM[i,outputs] = gate_TPM[j,:]
                break
    return expanded_gate_TPM
    

def remove_motor_sensor_effects(TPM,n_sensors=2,n_motors=2,n_nodes=4,states_convention = 'loli'):
    '''
    
        Inputs:
            
        Outputs:
            
    '''
    # forcing all sensors to be zero in the effect
    TPM[:,0:n_sensors] = np.ones(np.shape(TPM[:,0:n_sensors]))/2.

    # converting TPM to multidimensional representation for easier calling
    TPMmulti = pyphi.convert.to_multidimensional(TPM)

    # setting all output states to be identical to the state with motors being off (forcing the effect of motors to be null)
    no_motor_holi, no_motor_loli = get_states(n_nodes-2)
    no_motor_states = no_motor_holi if states_convention == 'holi' else no_motor_loli
    motor_holi, motor_loli = get_states(n_motors)
    motor_states = motor_holi if states_convention == 'holi' else motor_loli

    newTPM = copy.deepcopy(TPM)
    no_motor_activity = [0]*n_motors
    for state in no_motor_states:
        sensors = list(state[:n_sensors])
        hidden = list(state[n_sensors:])
        for motor_state in motor_states:
            full_state = tuple(sensors+list(motor_state)+hidden)
            if all(motor_state == no_motor_activity):
                out = TPMmulti[full_state]
            TPMmulti[full_state] = out

    TPM = pyphi.convert.to_2dimensional(TPMmulti)
    return TPM


def remove_motor_sensor_connections(cm,n_sensors=2,n_motors=2):
    '''
    
        Inputs:
            
        Outputs:
            
    '''
    # setting all connections to sensors to 0
    cm[:,0:n_sensors] = np.zeros(np.shape(cm[:,0:n_sensors]))
    # setting all connections from motors to 0
    cm[n_sensors:n_sensors+n_motors] = np.zeros(np.shape(cm[n_sensors:n_sensors+n_motors]))
    
    return cm

def genome2TPM(genome, n_nodes=8, n_sensors=2, n_motors=2, gate_type='deterministic',states_convention='loli',remove_sensor_motor_effects=False):
    '''
    Extracts the TPM from the genome output by mabe.
        Inputs:
            genome: np.array of the genome output from mabe (1 x n_codons)
            n_nodes:(maximal) number of nodes in the agent
            gate_type: string specifying the type of gates coded by the genome ('deterministic' or 'decomposable')
            states_convention: string specifying the convention used for ordering of the states ('holi' or 'loli')
            
        Outputs:
            TPM: The full state transition matrix for the agent (states x nodes)
            gate_TPMs: the TPM for each gate specified by the genome (which in turn specifies the full TPM)
            cm: Connectivity matrix (nodes x nodes) for the agent. 1's indicate there is a connection, 0's indicate the opposite
    '''
    max_gene_length = 400
    max_inputs = max_outputs = max_io = 4 # 4 inputs, 4 outputs per HMG
    if gate_type == 'deterministic':
        start_codon = 43
    elif gate_type == 'decomposable':
        start_codon = 50
    else:
        raise AttributeError("Unknown gate type.")

    print('Reading genome...')
    ixs = np.where(genome==start_codon)[0]
    gene_ixs = [ix for ix in ixs if genome[ix+1]==255-start_codon]

    genes = np.array([genome[ix:ix+max_gene_length] for ix in gene_ixs])
    n_genes = genes.shape[0]

    # -----------------------
    # read out genes
    # -----------------------
    # locations:
    # 2    number of inputs
    # 3    number of outputs
    # 4-7  possible inputs
    # 8-11 possible outputs
    # 12   TPM

    cm = np.zeros((n_nodes,n_nodes))
    full_TPM = np.zeros((2**n_nodes, n_nodes, n_genes))
    gate_TPMs = []

    for i, gene in zip(range(n_genes),genes):
        print('Gene: {}/{}'.format(i+1,n_genes))

        # Get gate's inputs and outputs
        n_inputs = gene[2] % max_inputs + 1
        n_outputs = gene[3] % max_outputs + 1
        raw_inputs = gene[4:4+n_inputs]
        raw_outputs = gene[8:8+n_outputs]
        inputs = gene[4:4+n_inputs] % n_nodes
        outputs = gene[8:8+n_outputs] % n_nodes

        # Get probabilities
        gate_TPM = np.zeros((2**n_inputs,n_outputs))
        for row in range(2**n_inputs):

            if start_codon == 50: # Decomposable
                start_locus = 12 + row * n_outputs + (max_io**4)
                raw_probability = gene[start_locus:start_locus+n_outputs]
                gate_TPM[row,:] = raw_probability/256 # or 255?

            else: # start_codon == 43 (Deterministic)
                start_locus = 12 + row * max_inputs # 12 or 13?
                raw_probability = gene[start_locus:start_locus+n_outputs]
                gate_TPM[row,:] = raw_probability % 2

        # Reduce gate's degenerate outputs (if there are)
        gate_TPM, outputs = reduce_degenerate_outputs(gate_TPM, outputs)
        gate_TPM, inputs = reduce_degenerate_inputs(gate_TPM, inputs, states_convention)

        gate_TPMs.append({'type': gate_type,
                          'ins': inputs,
                          'outs': outputs,
                          'logic': gate_TPM.tolist()})

        # Get list of all possible states from nodes
        cm[np.ix_(inputs,outputs)] = 1

        # Expand gate TPM
        full_TPM[:,:,i] = expand_gate_TPM(gate_TPM, inputs, outputs, n_nodes, states_convention)

    TPM = 1 - np.prod(1 - full_TPM,2)
    
    if remove_sensor_motor_effects:
        TPM = remove_motor_sensor_effects(TPM,n_sensors,n_motors,n_nodes)
        cm = remove_motor_sensor_connections(cm,n_sensors,n_motors)
    
    print('Done.')
    return TPM, gate_TPMs, cm
    
    
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

