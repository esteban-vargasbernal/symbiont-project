import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import random
import pandas as pd

import functions_symbionts as sym

import sys

from collections import Counter



df_all = pd.read_csv('Data/df_N_1000.csv')




scenarios = ['Mutualism', 'Predator-prey', 'Parasitism', 'Competition']
colors = {'Mutualism':"red",'Predator-prey':"blue", 'Parasitism':"green", 'Competition': "black"}

N_sim = 2
N_gen = 10

epsilon_cell_large = 0.001
epsilon_sym_large = 0.001 #u_sym*L

epsilon_cell_small = 0.00001
epsilon_sym_small = 0.00001 #u_sym*L

e_sym_tmp = np.linspace(-0.75,0.25,6)
e_cell_tmp = np.linspace(-0.75,0.25,6)
e_pairs = np.round(sym.pairwise_combinations(e_sym_tmp,e_cell_tmp),2)

n_args = np.shape(e_pairs)[0]

h_for_cells_ls = n_args*[0.25]
e_for_symbionts_ls = list(e_pairs[:,0])

h_for_symbionts_ls = n_args*[0.25]
e_for_cells_ls = list(e_pairs[:,1])


#df = df_all[ np.logical_and(df_all['epsilon_cell']==epsilon_cell_large, df_all['epsilon_sym']==epsilon_sym_large)]

df_sum = pd.DataFrame(columns=['theta_cell','theta_sym','scenario_start','scenario_end', 'Mutualism', 'Predator-prey', 'Parasitism', 'Competition'])

i = 0

for e_sym in e_for_symbionts_ls:

    for e_cell in e_for_cells_ls:
        
        for j in np.arange(N_sim):

            df = df_all[ np.logical_and(df_all['epsilon_cell']==epsilon_cell_large, df_all['epsilon_sym']==epsilon_sym_large)]
            df_tmp = df[np.round(df['e_sym_0'],2) == np.round(e_sym,2)]
            df_tmp = df_tmp[np.round(df_tmp['e_cell_0'],2)==np.round(e_cell,2)]
            df_tmp = df_tmp[df_tmp['sim']==j]

            theta_cell = df_tmp["theta_cell"].iloc[0]
            theta_sym = df_tmp['theta_sym'].iloc[0]

            scenario_start = df_tmp['scenario_start'].iloc[0]
            scenario_end = df_tmp['scenario_end'].iloc[0]
            
            list_tmp = list(df_tmp['scenario'])
            scenario_counts = {scena:list_tmp.count(scena) for scena in scenarios}
            
            mutualism = round(scenario_counts['Mutualism']/N_gen,2)
            predator_prey = round(scenario_counts['Predator-prey']/N_gen,2)
            parasitism = round(scenario_counts['Parasitism']/N_gen,2)
            competition = round(scenario_counts['Competition']/N_gen,2)
            print([theta_cell, theta_sym, scenario_start, scenario_end, mutualism, predator_prey, parasitism, competition])
            df_sum.loc[i] = [theta_cell, theta_sym, scenario_start, scenario_end, mutualism, predator_prey, parasitism, competition]
            i = i+1

            x = df_tmp[["theta_cell"]]
            y = df_tmp[['theta_sym']]
            
            if j == 0:
                plt.plot(x,y, '.', color = colors[scenario_start])
            else:
                plt.plot(x,y, '.', color = colors[scenario_start])
            
            #plt.plot(x.iloc[0], y.iloc[0],'.', color = colors[i])
            #plt.xlim(-2,2)
            #plt.ylim(-2,2)

print(df_sum)

df_sum.to_csv('Data/df_sum.csv', index=False)
ll = 1


plt.grid()
plt.xlabel('theta_host = e_sym + h_host')
plt.ylabel('theta_sym = e_host + h_sym')
#plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.tight_layout()
#plt.savefig('Figures/all_scenarios_'+str(N_gen)+'generations.png')
plt.show()