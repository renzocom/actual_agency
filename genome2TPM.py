import numpy as np
import os
from pathlib import Path
import scipy.io as sio

def genome2TPM(genome, n_nodes, gate_type='deterministic',states_convention='loli'):
    '''Extracts the full TPM (state by node) and gate TPMs from the genome (np.array).
    Obs: MABE uses loli (little endian) convention as Pyphi.'''

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
    print('Done.')
    return TPM, gate_TPMs, cm

def get_states(n_nodes, convention='loli'):
    '''Returns arrays (2**n, n) with all possible states according to holi (big endian) and
    loli (little endian) indexing conventions.'''

    states_holi = np.array(([list(('{:0'+str(n_nodes)+'d}').format(int(bin(x)[2:]))) for x in range(2**n_nodes)])).astype(int)
    states_loli = np.flip(states_holi,1)
    if convention=='loli':
        return states_loli
    else:
        return states_holi

def reduce_degenerate_outputs(gate_TPM, outputs):
    '''Reduces gate_TPM with degenerate outputs (e.g. outputs=[2,12,3,12] to outputs=[2,3,12]) by combining
    them with OR logic.'''

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

def expand_gate_TPM(gate_TPM, inputs, outputs, n_nodes, states_convention):
    '''Expands the gate TPM (2**n_inputs,n_outputs) to the full size TPM (2**n_nodes,n_nodes).'''

    full_states = get_states(n_nodes, convention='loli')
    gate_states = get_states(len(inputs), convention='loli')

    expanded_gate_TPM = np.zeros((2**n_nodes,n_nodes))

    for i in range(expanded_gate_TPM.shape[0]): # iterate through rows (inputs states)
        for j in range(gate_TPM.shape[0]):
            if np.all(gate_states[j,:]==full_states[i,inputs]):
                expanded_gate_TPM[i,outputs] = gate_TPM[j,:]
                break
    return expanded_gate_TPM

def reduce_degenerate_inputs(gate_TPM, inputs, states_convention):
    '''Reduces gate_TPM with degenerate inputs (e.g. inputs=[2,12,3,12] to inputs=[2,3,12]) by removing
    input states that are internally inconsistent (e.g. input_state(A=0,A=0,B=0) is consistent while
    input_state=(A=0,A=1,B=0) is inconsistent, thus removed).'''

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
    input_states = get_states(len(inputs),convention=states_convention)

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
