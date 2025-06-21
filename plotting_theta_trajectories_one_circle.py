import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
import random
import pandas as pd

from matplotlib.patches import Circle

import functions_symbionts as sym

import sys

from collections import Counter



#df_all = pd.read_csv('Data/df_N_1000.csv')

#df_all = pd.read_csv('Data/df_all_within_circle_only_competition.csv')

#df_angle = pd.read_csv('Data/df_angle_within_circle_only_competition.csv')

#df_all = pd.read_csv('Data/df_all_within_circle.csv')
#df_angle = pd.read_csv('Data/df_angle_within_circle.csv')

df_all_1 = pd.read_csv('Data/df_all_I.csv')
df_all_3 = pd.read_csv('Data/df_all_III.csv')
df_all = pd.concat([df_all_1,df_all_3])

df_angle_1 = pd.read_csv('Data/df_angle_I.csv')
df_angle_2 = pd.read_csv('Data/df_angle_III.csv')
df_angle = pd.concat([df_angle_1,df_angle_2])


scenarios = ['Mutualism', 'Predator-prey', 'Parasitism', 'Competition']
colors = {'Mutualism':"red",'Predator-prey':"blue", 'Parasitism':"green", 'Competition': "black"}


epsilon_cell = 0.001
epsilon_sym = 0.001 #u_sym*L

N_sim = 40
n_split = 2
N_gen_max = 500

e_for_symbionts_ls = np.linspace(-0.75,0.25,n_split)
e_for_cells_ls = np.linspace(-0.75,0.25,n_split)

h_sym = 0.25
h_cell = 0.25

e_sym = -0.75
e_cell = -0.75

theta_cell_0 = e_sym + h_cell 
theta_sym_0 = e_cell + h_sym 

### change the thetas below
theta_cell_0 = -0.5
theta_sym_0 =  -0.5

R = 0.01
circle = Circle((theta_cell_0, theta_sym_0), radius = R, color='blue', fill=False, linewidth=2)


                    
for j in np.arange(N_sim):

    df = df_all
    df_tmp = df[np.round(df['theta_sym_0'],2) == np.round(theta_sym_0,2)]
    df_tmp = df_tmp[np.round(df_tmp['theta_cell_0'],2)==np.round(theta_cell_0,2)]
    df_tmp = df_tmp[df_tmp['sim']==j]

    theta_cell = df_tmp["theta_cell"].iloc[0]
    theta_sym = df_tmp['theta_sym'].iloc[0]

    

    scenario_start = df_tmp['scenario_start'].iloc[0]
    scenario_end = df_tmp['scenario_end'].iloc[0]
    
    x = df_tmp[["theta_cell"]]
    y = df_tmp[['theta_sym']]
    
    if j == 0:
        if (e_sym, e_cell) in [(np.max(e_for_symbionts_ls), np.max(e_for_cells_ls)), 
                            (np.min(e_for_symbionts_ls), np.max(e_for_cells_ls)),
                            (np.max(e_for_symbionts_ls), np.min(e_for_cells_ls)),
                            (np.min(e_for_symbionts_ls), np.min(e_for_cells_ls))]:
            plt.plot(x.iloc[0],y.iloc[0], '.', color = colors[scenario_start], label = scenario_start)
            plt.plot(x,y, '-', alpha = 0.1, color = colors[scenario_start])
        else:
            plt.plot(x.iloc[0],y.iloc[0], '.', color = colors[scenario_start])
            plt.plot(x,y, '-', alpha = 0.1, color = colors[scenario_start])
    else:
        plt.plot(x,y, '-', alpha = 0.1, color = colors[scenario_start])
    

    plt.plot(x.iloc[-1], y.iloc[-1],'*',alpha = 0.5, linewidth =3,color = colors[scenario_start])
    plt.gca().add_patch(circle)


df_angle_tmp = df_angle[np.round(df_angle['theta_sym'],2) == np.round(theta_sym_0,2)]
df_angle_tmp = df_angle_tmp[np.round(df_angle_tmp['theta_cell'],2)==np.round(theta_cell_0,2)]
df_angle_tmp = df_angle_tmp[df_angle_tmp['generation'] < (N_gen_max-1) ]

angles_v = df_angle_tmp['angle']

print(angles_v)

mean_x = np.mean( np.cos(angles_v))
mean_y = np.mean( np.sin(angles_v))
#print(mean_angle)
sd_angle = np.std(angles_v)
sd_x = np.std(np.cos(angles_v))
sd_y = np.std(np.sin(angles_v))
sd_xy = np.sqrt(sd_x**2 + sd_y**2)
print(sd_xy)
range_angle = np.max(angles_v) - np.min(angles_v)
#plt.hlines(y = 0, xmin=-1, xmax =2, color = "black")
#plt.vlines(x = 0, ymin =-1, ymax = 1.2, color = "black")
#plt.xlim(-1,2)
#plt.ylim(-1,1.2)

plt.grid()
plt.locator_params(axis='x', nbins=5)
plt.xlabel(r'$\theta_{\text{host}} = e_{\text{sym}} + h_{\text{host}}$', fontsize = 16)
plt.ylabel(r'$\theta_{\text{sym}} = e_{\text{host}} + h_{\text{sym}}$', fontsize = 16)
plt.arrow(theta_cell_0, theta_sym_0, R/(1+sd_xy)*mean_x, R/(1+sd_xy)*mean_y, width=0.0005)
plt.title(r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym), fontsize = 18)
plt.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0, title = 'Scenario of the \n initial condition')
plt.tight_layout()
plt.savefig('Figures/theta_trajectories_one_circlet_'+ scenario_start+'.png')
plt.show()