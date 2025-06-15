import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import random
import pandas as pd
import multiprocessing
import os
import time 
import itertools

import sys




def death_cells(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_cell):
    
    #np.random.seed(123)
    
    # kill cells
    x_alive_copy = copy.deepcopy(x_alive)
    
    alive_cell_ids = np.array(np.where(x_alive_copy['cell']==1)).transpose()
    N_alive = alive_cell_ids.shape[0]
    death_cell_rows = np.random.choice(np.arange(N_alive), math.floor(D_cell*N_alive), replace = False)
    dead_cells = alive_cell_ids[death_cell_rows,:]
    
    x_alive_copy['cell'][dead_cells[:,0]] = 0
    x_alive_copy['sym_in'][dead_cells[:,0], :] = 0 
        
    return(x_alive_copy)

#################

def death_sym_in(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_sym):
    
    #np.random.seed(123)
    
    # kill symbionts inside cells
    x_alive_copy = copy.deepcopy(x_alive)
    
    alive_cell_ids = np.array(np.where(x_alive_copy['sym_in']==1)).transpose()
    N_alive = alive_cell_ids.shape[0]
    death_cell_rows = np.random.choice(np.arange(N_alive), math.floor(D_sym*N_alive), replace = False)
    dead_cells = alive_cell_ids[death_cell_rows,:]
    
    x_alive_copy['sym_in'][dead_cells[:,0], dead_cells[:,1]] = 0 # symbionts in dead cells also die?
        
    return(x_alive_copy)

#################

def death_sym_out(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_sym):

    #np.random.seed(123)
    
    # kill symbionts inside cells
    x_alive_copy = copy.deepcopy(x_alive)
    
    alive_cell_ids = np.array(np.where(x_alive_copy['sym_out']==1)).transpose()
    N_alive = alive_cell_ids.shape[0]
    death_cell_rows = np.random.choice(np.arange(N_alive), math.floor(D_sym*N_alive), replace = False)
    dead_cells = alive_cell_ids[death_cell_rows,:]

    x_alive_copy['sym_out'][0, dead_cells[:,1]] = 0 # symbionts in dead cells also die?
        
    return(x_alive_copy)

#################

def update_traits_after_death(N,c,M,x_alive_new,x_cell,x_sym_in,x_sym_out):
    
    #np.random.seed(123)
    
    x_cell_copy = copy.deepcopy(x_cell)
    x_sym_in_copy = copy.deepcopy(x_sym_in)
    x_sym_out_copy = copy.deepcopy(x_sym_out)
    
    s_cell = x_cell_copy['s']
    h_cell = x_cell_copy['h']
    e_cell = x_cell_copy['e']
    
    s_sym_in = x_sym_in_copy['s']
    h_sym_in = x_sym_in_copy['h']
    e_sym_in = x_sym_in_copy['e']
    
    s_sym_out = x_sym_out_copy['s']
    h_sym_out = x_sym_out_copy['h']
    e_sym_out = x_sym_out_copy['e']
    
    
    s_cell = np.multiply(s_cell, x_alive_new['cell']) + 1 - x_alive_new['cell']
    h_cell = np.multiply(h_cell, x_alive_new['cell'])
    e_cell = np.multiply(e_cell, x_alive_new['cell'])
    
    s_sym_in = np.multiply(s_sym_in, x_alive_new['sym_in']) + 1 - x_alive_new['sym_in']
    h_sym_in = np.multiply(h_sym_in, x_alive_new['sym_in'])
    e_sym_in = np.multiply(e_sym_in, x_alive_new['sym_in'])
    
    s_sym_out = np.multiply(s_sym_out, x_alive_new['sym_out']) + 1 - x_alive_new['sym_out']
    h_sym_out = np.multiply(h_sym_out, x_alive_new['sym_out'])
    e_sym_out = np.multiply(e_sym_out, x_alive_new['sym_out'])
    
    x_cell_new = dict(s = s_cell, h = h_cell, e = e_cell)
    x_sym_in_new = dict(s = s_sym_in, h = h_sym_in, e = e_sym_in)
    x_sym_out_new = dict(s = s_sym_out, h = h_sym_out, e = e_sym_out)

    return(x_cell_new, x_sym_in_new, x_sym_out_new)


#################


def fitness(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out):
    
    #np.random.seed(123)
    
    k_sym = np.sum(x_alive['sym_in'],axis =1).reshape(N,1)
    
    fitness_cell = x_cell['s'] +  np.multiply(x_cell['h'],k_sym) + np.sum(x_sym_in['e'], axis = 1).reshape(N,1)
    fitness_cell = np.maximum(0,fitness_cell)
    fitness_cell = np.multiply(fitness_cell, x_alive['cell']) + 1 - x_alive['cell']
    
    fitness_sym_in = x_sym_in['s'] + x_sym_in['h'] + np.dot(x_cell['e'], np.ones((1,c)))
    fitness_sym_in = np.maximum(0, fitness_sym_in)
    fitness_sym_in = np.multiply(fitness_sym_in,x_alive['sym_in']) + 1 - x_alive['sym_in']
    
    fitness_sym_out = x_sym_out['s']
    fitness_sym_out = np.maximum(0, fitness_sym_out)
    fitness_sym_out = np.multiply(fitness_sym_out,x_alive['sym_out']) + 1 - x_alive['sym_out']
    
    x_fitness = dict(cell = fitness_cell, sym_in = fitness_sym_in, sym_out = fitness_sym_out)
    
    return(x_fitness)


#################

def birth_cell(N,c,M,x_alive,x_fitness, x_cell, x_sym_in):
    
    #np.random.seed(123)
    x_alive_copy = copy.deepcopy(x_alive)
    x_cell_copy = copy.deepcopy(x_cell)
    x_sym_in_copy = copy.deepcopy(x_sym_in)
    
    if sum(x_fitness['cell']) != 0:
        
        prob = (x_fitness['cell']/sum(x_fitness['cell'])).reshape(-1)

        dead_cell_ids = np.array(np.where(x_alive['cell']==0)).transpose()

        birth_cell_ids = np.random.choice(np.arange(N), dead_cell_ids.shape[0], replace = True, p = prob)



        # update traits


        x_alive_copy['cell'][list(dead_cell_ids[:,0])] = x_alive_copy['cell'][list(birth_cell_ids)] 
        x_alive_copy['sym_in'][list(dead_cell_ids[:,0]), :] = x_alive_copy['sym_in'][list(birth_cell_ids),:] 

        x_cell_copy['s'][list(dead_cell_ids[:,0])] = x_cell_copy['s'][list(birth_cell_ids)]
        x_cell_copy['h'][list(dead_cell_ids[:,0])] = x_cell_copy['h'][list(birth_cell_ids)]
        x_cell_copy['e'][list(dead_cell_ids[:,0])] = x_cell_copy['e'][list(birth_cell_ids)]


        x_sym_in_copy['s'][list(dead_cell_ids[:,0]), :] = x_sym_in_copy['s'][list(birth_cell_ids), :]
        x_sym_in_copy['h'][list(dead_cell_ids[:,0]), :] = x_sym_in_copy['h'][list(birth_cell_ids), :]
        x_sym_in_copy['e'][list(dead_cell_ids[:,0]), :] = x_sym_in_copy['e'][list(birth_cell_ids), :]


    return(x_alive_copy, x_cell_copy, x_sym_in_copy)

#################

def birth_sym_in(N,c,M,x_alive_pre,x_alive_current,x_fitness, x_sym_in):
    
    #np.random.seed(123)
    
    prob = np.dot(np.diag(1/np.sum(x_fitness['sym_in'], axis = 1)),x_fitness['sym_in'])
    
    
    alive_cell_ids = np.array(np.where(x_alive_pre['cell']==1)).transpose()
    
    x_alive_copy = copy.deepcopy(x_alive_current)
    x_sym_in_copy = copy.deepcopy(x_sym_in)
    
    for i in alive_cell_ids[:,0]:
        
        if np.sum(x_fitness['sym_in'][i,:]) != 0:
                  
            dead_sym_in_ids = np.array(np.where(x_alive_pre['sym_in'][i,:]==0))[0]

            birth_sym_in_ids = np.random.choice(np.arange(c), dead_sym_in_ids.shape[0], replace = True, p = prob[i,:])

            x_alive_copy['sym_in'][i,list(dead_sym_in_ids)] = x_alive_copy['sym_in'][i,list(birth_sym_in_ids)] 

            x_sym_in_copy['s'][i,list(dead_sym_in_ids)] = x_sym_in_copy['s'][i,list(birth_sym_in_ids)]
            x_sym_in_copy['h'][i,list(dead_sym_in_ids)] = x_sym_in_copy['h'][i,list(birth_sym_in_ids)]
            x_sym_in_copy['e'][i,list(dead_sym_in_ids)] = x_sym_in_copy['e'][i,list(birth_sym_in_ids)]


    return(x_alive_copy, x_sym_in_copy)


#################

def birth_sym_out(N,c,M,x_alive_pre,x_fitness, x_sym_out):
    
    #np.random.seed(123)
    x_alive_copy = copy.deepcopy(x_alive_pre)
    x_sym_out_copy = copy.deepcopy(x_sym_out)
    
    
            
    if np.sum(x_fitness['sym_out']) != 0:
        
        prob = (x_fitness['sym_out']/np.sum(x_fitness['sym_out'])).reshape(-1)
        
        dead_sym_out_ids = np.array(np.where(x_alive_pre['sym_out']==0))[1]

        birth_sym_out_ids = np.random.choice(np.arange(M), dead_sym_out_ids.shape[0], replace = True, p = prob)

        x_alive_copy['sym_out'][0,list(dead_sym_out_ids)] = x_alive_copy['sym_out'][0,list(birth_sym_out_ids)] 

        x_sym_out_copy['s'][0,list(dead_sym_out_ids)] = x_sym_out_copy['s'][0,list(birth_sym_out_ids)]
        x_sym_out_copy['h'][0,list(dead_sym_out_ids)] = x_sym_out_copy['h'][0,list(birth_sym_out_ids)]
        x_sym_out_copy['e'][0,list(dead_sym_out_ids)] = x_sym_out_copy['e'][0,list(birth_sym_out_ids)]


    return(x_alive_copy, x_sym_out_copy)

#################

def swap(N,c,M,x_alive,x_sym_in,x_sym_out,W_in, W_out):
    
    #np.random.seed(123)
    
    x_alive_copy = copy.deepcopy(x_alive)
    x_sym_in_copy = copy.deepcopy(x_sym_in)
    x_sym_out_copy = copy.deepcopy(x_sym_out)
    
    W_in_list = []
    W_out_list = []
    
    alive_cell_ids = np.array(np.where(x_alive_copy['cell']==1)).transpose()
    N_alive = alive_cell_ids.shape[0]
    
    while len(W_in_list) < 2*W_in:
        
       
        i_pair_ids = np.random.choice(np.arange(N_alive), 2, replace = False)
        i_pair = list(alive_cell_ids[i_pair_ids,0])
    
        j_pair = list(np.random.choice(np.arange(c),2, replace = True))

        if( ([i_pair[0],j_pair[0]] not in W_in_list) and ([i_pair[1],j_pair[1]] not in W_in_list) ):
            
            
            W_in_list.append([i_pair[0],j_pair[0]])
            W_in_list.append([i_pair[1],j_pair[1]])
            
            x_alive_copy['sym_in'][i_pair[0],j_pair[0]] = x_alive['sym_in'][i_pair[1],j_pair[1]] 
            
            x_sym_in_copy['s'][i_pair[0],j_pair[0]] = x_sym_in['s'][i_pair[1],j_pair[1]]
            x_sym_in_copy['h'][i_pair[0],j_pair[0]] = x_sym_in['h'][i_pair[1],j_pair[1]]
            x_sym_in_copy['e'][i_pair[0],j_pair[0]] = x_sym_in['e'][i_pair[1],j_pair[1]]

            x_alive_copy['sym_in'][i_pair[1],j_pair[1]] = x_alive['sym_in'][i_pair[0],j_pair[0]] 
            
            x_sym_in_copy['s'][i_pair[1],j_pair[1]] = x_sym_in['s'][i_pair[0],j_pair[0]]
            x_sym_in_copy['h'][i_pair[1],j_pair[1]] = x_sym_in['h'][i_pair[0],j_pair[0]]
            x_sym_in_copy['e'][i_pair[1],j_pair[1]] = x_sym_in['e'][i_pair[0],j_pair[0]]

    while len(W_out_list) < W_out:
        
        i_in_id = np.random.choice(np.arange(N_alive))
        i_in = alive_cell_ids[i_in_id,0]
        
        j_in = np.random.choice(np.arange(c))
        j_out = np.random.choice(np.arange(M))
        
        if( ([i_in,j_in] not in W_in_list) and (j_out not in W_out_list)):
                        
            W_out_list.append(j_out)
            
            x_alive_copy['sym_in'][i_in,j_in] = x_alive['sym_out'][0,j_out] 
            x_alive_copy['sym_out'][0,j_out] = x_alive['sym_in'][i_in,j_in] 
            
            x_sym_in_copy['s'][i_in,j_in] = x_sym_out['s'][0,j_out]
            x_sym_in_copy['h'][i_in,j_in] = x_sym_out['h'][0,j_out]
            x_sym_in_copy['e'][i_in,j_in] = x_sym_out['e'][0,j_out]
            
            x_sym_out_copy['s'][0,j_out] = x_sym_in['s'][i_in,j_in]
            x_sym_out_copy['h'][0,j_out] = x_sym_in['h'][i_in,j_in]
            x_sym_out_copy['e'][0,j_out] = x_sym_in['e'][i_in,j_in]

    return(x_alive_copy, x_sym_in_copy, x_sym_out_copy)
        

#################

def mutation(N,c,M,x_alive, x_cell, x_sym_in,x_sym_out,x_mut):
    
    epsilon_cell = x_mut['epsilon_cell']
    epsilon_sym_in = x_mut['epsilon_sym_in']
    epsilon_sym_out = x_mut['epsilon_sym_out']
    
    gamma_cell = x_mut['gamma_cell']
    gamma_sym_in = x_mut['gamma_sym_in']
    gamma_sym_out = x_mut['gamma_sym_out']
    
    
    x_alive_copy = copy.deepcopy(x_alive)
    x_cell_copy = copy.deepcopy(x_cell)
    x_sym_in_copy = copy.deepcopy(x_sym_in)
    x_sym_out_copy = copy.deepcopy(x_sym_out)
    
    # mutations of hosts
    
    
    alive_cell = np.where(x_alive_copy['cell']==1)
    N_alive_cell = len(list(alive_cell[0]))
    
    new_mu_mut_cell = epsilon_cell*N_alive_cell
    
    n_mut_cell = min(np.random.poisson(new_mu_mut_cell,1)[0], N_alive_cell)
    
    ids_alive_cell = np.random.choice(np.arange(N_alive_cell), n_mut_cell, replace = False)
    i_mutants = alive_cell[0][ids_alive_cell]
    
    
    #print('Number of mutations of hosts = ', n_mut_cell)
    
    delta_cell = np.random.uniform(-gamma_cell, gamma_cell, n_mut_cell*3).reshape(n_mut_cell,3)
    
    for i in np.arange(n_mut_cell):
    
        #print('')
        #print('i_mutants = ', i_mutants[i])
        #print('')
        
        x_cell_copy['s'][i_mutants[i],0] = x_cell['s'][i_mutants[i],0] + delta_cell[i,0]
        x_cell_copy['h'][i_mutants[i],0] = x_cell['h'][i_mutants[i],0] + delta_cell[i,1]
        x_cell_copy['e'][i_mutants[i],0] = x_cell['e'][i_mutants[i],0] + delta_cell[i,2]
    
    
    # mutation inside hosts
    
    alive_sym_in = np.where(x_alive_copy['sym_in']==1)
    N_alive_in = len(list(alive_sym_in[0]))
    
    new_mu_mut_in = epsilon_sym_in*N_alive_in
    n_mut_in = min(np.random.poisson(new_mu_mut_in,1)[0], N_alive_in)
    
    ids_alive_in = np.random.choice(np.arange(N_alive_in), n_mut_in, replace = False)
    i_mutants = alive_sym_in[0][ids_alive_in]
    j_mutants = alive_sym_in[1][ids_alive_in]
    
    #print('Number of mutations inside hosts = ', n_mut_in)
    
    delta_in = np.random.uniform(-gamma_sym_in,gamma_sym_in,n_mut_in*3).reshape(n_mut_in,3)
    
    for i in np.arange(n_mut_in):
        i_trait = np.random.choice(np.arange(3))
    
  
        #print('')
        #print('i_mutants = ', i_mutants[i])
        #print('')
        #print('j_mutants = ', j_mutants[i])
        #print('')
        
        x_sym_in_copy['s'][i_mutants[i],j_mutants[i]] =  x_sym_in['s'][i_mutants[i],j_mutants[i]] + delta_in[i,0]
        x_sym_in_copy['h'][i_mutants[i],j_mutants[i]] =  x_sym_in['h'][i_mutants[i],j_mutants[i]] + delta_in[i,1]
        x_sym_in_copy['e'][i_mutants[i],j_mutants[i]] =  x_sym_in['e'][i_mutants[i],j_mutants[i]] + delta_in[i,2]
    
    
    # mutations out of hosts
    
    alive_sym_out = np.where(x_alive_copy['sym_out']==1)
    N_alive_out = len(list(alive_sym_out[0]))
    
    new_mu_mut_out = epsilon_sym_out*N_alive_out
    n_mut_out = min(np.random.poisson(new_mu_mut_out,1)[0],N_alive_out)
    
    ids_alive_out = np.random.choice(np.arange(N_alive_out), n_mut_out, replace = False)
    i_mutants_out = alive_sym_out[1][ids_alive_out]

    
    #print('Number of mutations outside hosts = ', n_mut_out)
    
    delta_out = np.random.uniform(-gamma_sym_out,gamma_sym_out,n_mut_out*3).reshape(n_mut_out,3)
    
    for i in np.arange(n_mut_out):
            

        #print('')
        #print('i_mutants_out = ', i_mutants_out[i])
        #print('')

        x_sym_out_copy['s'][0,i_mutants_out[i]] = x_sym_out_copy['s'][0,i_mutants_out[i]] + delta_out[i,0]
        x_sym_out_copy['h'][0,i_mutants_out[i]] = x_sym_out_copy['h'][0,i_mutants_out[i]] + delta_out[i,1]
        x_sym_out_copy['e'][0,i_mutants_out[i]] = x_sym_out_copy['e'][0,i_mutants_out[i]] + delta_out[i,2]
        
    
    return(x_cell_copy, x_sym_in_copy, x_sym_out_copy)


#################

def one_generation(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_cell,D_sym,W_in, W_out, x_mut):
    
    #np.random.seed(123)

    # Death
    
    x_alive1 = death_cells(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_cell)
    x_alive2 = death_sym_in(N,c,M,x_alive1,x_cell,x_sym_in,x_sym_out,D_sym)
    x_alive3 = death_sym_out(N,c,M,x_alive2,x_cell,x_sym_in,x_sym_out,D_sym)
    
    x_cell_new, x_sym_in_new, x_sym_out_new = update_traits_after_death(N,c,M,x_alive3,x_cell,x_sym_in,x_sym_out)
    
    
    # Birth
    
    x_fitness = fitness(N,c,M,x_alive3,x_cell_new,x_sym_in_new,x_sym_out_new)
    #print(x_fitness)
    x_alive4, x_cell_new2, x_sym_in_new2 = birth_cell(N,c,M,x_alive3,x_fitness, x_cell_new, x_sym_in_new)
    x_alive5, x_sym_in_new3 = birth_sym_in(N,c,M,x_alive3,x_alive4,x_fitness, x_sym_in_new2)
    x_alive6, x_sym_out_new2 = birth_sym_out(N,c,M,x_alive5,x_fitness, x_sym_out_new)

    
    # Swap
    
    x_alive7,x_sym_in_new4, x_sym_out_new3 = swap(N,c,M,x_alive6,x_sym_in_new3,x_sym_out_new2,W_in, W_out)
    

    # mutation
    
    x_cell_new3, x_sym_in_new5, x_sym_out_new4 = mutation(N,c,M,x_alive7, x_cell_new2, x_sym_in_new4, x_sym_out_new3,x_mut)

    
    return(x_alive7, x_cell_new3, x_sym_in_new5, x_sym_out_new4)



#################

def mean_or_zero(data):
       if len(data) == 0:
           return 0
       else:
           return np.mean(data)

#################

def model_symbionts_with_mutation(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_cell,D_sym,W_in, W_out, x_mut, N_gen):
    
    x_alive_tmp = copy.deepcopy(x_alive)
    x_cell_tmp =  copy.deepcopy(x_cell)
    x_sym_in_tmp = copy.deepcopy(x_sym_in)
    x_sym_out_tmp = copy.deepcopy(x_sym_out)
    
    col_names = ['generation', 'alive_cell', 'alive_sym_in', 'alive_sym_out',
           'sym_in_dead',
           'sym_out_dead',
           'fitness_cell', 'fitness_sym_in', 'fitness_sym_out',
                's_cell', 'h_cell', 'e_cell',
                's_sym_in', 'h_sym_in', 'e_sym_in',
                'theta_cell', 'theta_sym','scenario']
    
    df = pd.DataFrame(columns = col_names)

    for i in np.arange(N_gen):
        
        generation = i

        alive_cell = len(np.where(x_alive_tmp['cell']==1)[0])
        alive_sym_in = len(np.where(x_alive_tmp['sym_in']==1)[0])
        #print('alive sym',alive_sym_in)
        alive_sym_out = len(np.where(x_alive_tmp['sym_out']==1)[0])

        sym_in_dead = len(np.where(x_sym_in_tmp['h']==0)[0])

        sym_out_dead = len(np.where(x_sym_out_tmp['h']==0)[0])

        fit = fitness(N,c,M,x_alive_tmp,x_cell_tmp,x_sym_in_tmp,x_sym_out_tmp)

        fitness_cell = mean_or_zero(fit['cell'][np.where(x_alive_tmp['cell']==1)])

        fitness_sym_in =  mean_or_zero(fit['sym_in'][np.where(x_alive_tmp['sym_in']==1)])
        
        fitness_sym_out = mean_or_zero(fit['sym_out'][np.where(x_alive_tmp['sym_out']==1)])
        
        s_sym_in = mean_or_zero(x_sym_in_tmp['s'][np.where(x_alive_tmp['sym_in']==1)])
        h_sym_in = mean_or_zero(x_sym_in_tmp['h'][np.where(x_alive_tmp['sym_in']==1)])
        e_sym_in = mean_or_zero(x_sym_in_tmp['e'][np.where(x_alive_tmp['sym_in']==1)])

        s_cell = mean_or_zero(x_cell_tmp['s'][np.where(x_alive_tmp['cell']==1)])
        h_cell = mean_or_zero(x_cell_tmp['h'][np.where(x_alive_tmp['cell']==1)])
        e_cell = mean_or_zero(x_cell_tmp['e'][np.where(x_alive_tmp['cell']==1)])

        theta_cell = e_sym_in + h_cell
        theta_sym = e_cell + h_sym_in

        if theta_cell >= 0 and theta_sym >= 0:
            scenario = "Mutualism"   
        if theta_cell > 0 and theta_sym < 0:
            scenario = "Predator-prey"    
        if theta_cell < 0 and theta_sym > 0:
            scenario = "Parasitism" 
        if theta_cell < 0 and theta_sym < 0:
            scenario = "Competition"       


        data_tmp = [generation, alive_cell, alive_sym_in, alive_sym_out,
                    sym_in_dead,
                    sym_out_dead,
                    fitness_cell, fitness_sym_in, fitness_sym_out,
                    s_cell, h_cell, e_cell,
                    s_sym_in, h_sym_in, e_sym_in,
                    theta_cell, theta_sym, scenario]
        
        df.loc[i] = data_tmp
        
        #if i%100 ==0: 
            #print('')
            #print('Generation number ', i)
            #print('')
            
        
        #np.random.seed(123 + 100*i)
        x_alive_tmp,x_cell_tmp,x_sym_in_tmp,x_sym_out_tmp =one_generation(N,c,M,x_alive_tmp,x_cell_tmp,x_sym_in_tmp,x_sym_out_tmp,D_cell,D_sym,W_in, W_out, x_mut)
        
    df['scenario_start'] = df['scenario'].iloc[0]
    df['scenario_end'] = df['scenario'].iloc[-1] 
    df['epsilon_cell'] = x_mut['epsilon_cell']
    df['epsilon_sym'] = x_mut['epsilon_sym_in']  
    return( df)

#################


def many_simulations_with_mutations(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_cell,D_sym,W_in, W_out, x_mut, N_gen,N_sim):
    
    col_names = ['generation', 'alive_cell', 'alive_sym_in', 'alive_sym_out',
           'sym_in_dead',
           'sym_out_dead',
           'fitness_cell', 'fitness_sym_in', 'fitness_sym_out',
            's_cell', 'h_cell', 'e_cell',
            's_sym_in', 'h_sym_in', 'e_sym_in',
                'theta_cell', 'theta_sym', 'scenario',
                'scenario_start','scenario_end',
                'epsilon_cell', 'epsilon_sym',
                'e_cell_0', 'e_sym_0']
    
    df = pd.DataFrame(columns = col_names)

    
    for j in np.arange(N_sim):
        print('')
        print('######################')
        print('Simulation number = ',j)
        print('######################')
        print('')
        df_tmp = model_symbionts_with_mutation(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_cell,D_sym,W_in, W_out, x_mut, N_gen)
        df_tmp['sim'] = N_gen*[j]
        df_tmp['e_cell_0'] =  np.round(df_tmp['e_cell'].iloc[0],2)
        df_tmp['e_sym_0'] = np.round(df_tmp['e_sym_in'].iloc[0],2)
        df = pd.concat([df,df_tmp])
    
    return(df)
        
def pairwise_combinations(vector1, vector2):
    return np.array(list(itertools.product(vector1, vector2)))
