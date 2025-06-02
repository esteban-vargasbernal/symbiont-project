import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import random
import pandas as pd

import functions_symbionts as sym

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)



#np.random.seed(123)
np.random.seed()
N = 200
c = 20
M = 100



D_cell = 0.4
D_sym = 0.4

W_in = 5
W_out= 5

N_sim = 5
N_gen = 1501



s_for_cells  = 1
h_for_cells = 1.95
e_for_symbionts = -2




s_for_symbionts = 2
h_for_symbionts = 1
e_for_cells = 1


# mutation parameters

epsilon_cell = 0.001
epsilon_sym_in = 0
epsilon_sym_out = 0

gamma_cell = 1
gamma_sym_in = 1
gamma_sym_out = 1


    
x_mut = dict(epsilon_cell = epsilon_cell,
    epsilon_sym_in = epsilon_sym_in,
    epsilon_sym_out = epsilon_sym_out, 
    gamma_cell = gamma_cell,
    gamma_sym_in = gamma_sym_in,
    gamma_sym_out = gamma_sym_out)


# everybody is alive at the beginning
alive_cell = np.ones((N,1))
alive_sym_in = np.zeros((N,c))
alive_sym_out = np.ones((1,M))

# proportion of type I symbionts
#p_large_h = 0.5


x_alive = dict(cell = alive_cell, sym_in = alive_sym_in, sym_out = alive_sym_out)

# all cells have the same traits

s_cell = s_for_cells*np.ones((N,1))
h_cell = h_for_cells*np.ones((N,1))
e_cell = e_for_cells*np.ones((N,1))

x_cell = dict(s = s_cell, h = h_cell, e = e_cell)

s_sym_in = s_for_symbionts*np.zeros((N,c))
h_sym_in = h_for_symbionts*np.zeros((N,c))
e_sym_in = e_for_symbionts*np.zeros((N,c))


x_sym_in = dict(s = s_sym_in, h = h_sym_in, e = e_sym_in)


s_sym_out = s_for_symbionts*np.ones((1,M))
h_sym_out = h_for_symbionts*np.ones((1,M))
e_sym_out = e_for_symbionts*np.ones((1,M))

x_sym_out = dict(s = s_sym_out, h = h_sym_out, e = e_sym_out)


df = sym.plotting_with_mutations(N,c,M,x_alive,x_cell,x_sym_in,x_sym_out,D_cell,D_sym,W_in, W_out, x_mut, N_gen,N_sim,s_for_symbionts, h_for_symbionts, e_for_symbionts,s_for_cells, h_for_cells, e_for_cells)

