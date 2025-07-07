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

#df_all = pd.read_csv('Data/df_small_N_1000_3.csv')


scenarios = ['Mutualism', 'Predator-prey', 'Parasitism', 'Competition']
colors = {'Mutualism':"red",'Predator-prey':"blue", 'Parasitism':"green", 'Competition': "black"}

N_sim = 5
N_gen = 500

epsilon_cell_large = 0.001
epsilon_sym_large = 0.001 #u_sym*L

epsilon_cell_small = 0.00001
epsilon_sym_small = 0.00001 #u_sym*L

n_split = 6

theta_for_symbionts_ls = np.linspace(-0.5,0.5,n_split)
theta_for_cells_ls = np.linspace(-0.5,0.5,n_split)



h_for_cells_ls = n_split*[0]
h_for_symbionts_ls = n_split*[0]




for epsilon_cell in [epsilon_cell_large, epsilon_cell_small]:
    
    for epsilon_sym in [epsilon_sym_large, epsilon_sym_small]:

        for theta_sym in theta_for_symbionts_ls[[2,3]]:

                for theta_cell in theta_for_cells_ls[[2,3]]:
                    
                    for j in np.arange(N_sim):

                        df = df_all[ np.logical_and(df_all['epsilon_cell']==epsilon_cell, df_all['epsilon_sym']==epsilon_sym)]
                        df_tmp = df[np.round(df['theta_sym_0'],2) == np.round(theta_sym,2)]
                        df_tmp = df_tmp[np.round(df_tmp['theta_cell_0'],2)==np.round(theta_cell,2)]
                        df_tmp = df_tmp[df_tmp['sim']==j]

                        theta_cell = df_tmp["theta_cell"].iloc[0]
                        theta_sym = df_tmp['theta_sym'].iloc[0]

                        scenario_start = df_tmp['scenario_start'].iloc[0]
                        scenario_end = df_tmp['scenario_end'].iloc[0]
                       
                        x = df_tmp[["theta_cell"]]
                        y = df_tmp[['theta_sym']]
                        
                        if j == 0:
                            if (theta_sym, theta_cell) in [(np.max(theta_for_symbionts_ls), np.max(theta_for_cells_ls)), 
                                                (np.min(theta_for_symbionts_ls), np.max(theta_for_cells_ls)),
                                                (np.max(theta_for_symbionts_ls), np.min(theta_for_cells_ls)),
                                                (np.min(theta_for_symbionts_ls), np.min(theta_for_cells_ls))]:
                                plt.plot(x.iloc[0],y.iloc[0], 'D', color = colors[scenario_start], label = scenario_start)
                                plt.plot(x,y, '-', alpha = 0.1, color = colors[scenario_start])
                            else:
                                plt.plot(x.iloc[0],y.iloc[0], 'D', color = colors[scenario_start])
                                plt.plot(x,y, '-', alpha = 0.1, color = colors[scenario_start])
                        else:
                            plt.plot(x,y, '-', alpha = 0.1, color = colors[scenario_start])
                        
                        plt.plot(x.iloc[-1], y.iloc[-1],'*',alpha = 0.5, linewidth =3,color = colors[scenario_start])
                        


        plt.hlines(y = 0, xmin=-1, xmax =2, color = "black")
        plt.vlines(x = 0, ymin =-1, ymax = 1.2, color = "black")
        plt.xlim(-1,2)
        plt.ylim(-1,1.2)

        plt.grid()
        plt.xlabel(r'$\theta_{\text{host}} = e_{\text{sym}} + h_{\text{host}}$', fontsize = 16)
        plt.ylabel(r'$\theta_{\text{sym}} = e_{\text{host}} + h_{\text{sym}}$', fontsize = 16)
        plt.title(r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym), fontsize = 18)
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, title = 'Scenario of the \n initial condition')
        plt.tight_layout()
        plt.savefig('Figures/theta_trajectories_epsilon_host_'+ str(epsilon_cell)+ '_epsilon_sym_'+str(epsilon_sym)+'.png')
        plt.show()






extract_feature = 'w_cell' # 'alive_cell', 'alive_sym_in', 'alive_sym_out', 's_cell'
N_sim_max = 1
theta_list = np.linspace(-0.5,0.5,n_split)
theta_cell = theta_list[0]
theta_sym = theta_list[0]

print(df_all)
df = df_all[ np.logical_and(df_all['epsilon_cell']==epsilon_cell, df_all['epsilon_sym']==epsilon_sym)]
df_tmp = df[np.round(df['theta_sym_0'],2) == np.round(theta_sym,2)]
df_tmp = df_tmp[np.round(df_tmp['theta_cell_0'],2)==np.round(theta_cell,2)]
df_tmp = df_tmp[df_tmp['sim']<N_sim_max]


df_plot = df_tmp.pivot(columns = 'sim', index = 'generation')[extract_feature]
df_plot.plot()
plt.ylabel(extract_feature, fontsize =16)
plt.title(r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym) + r'$, \theta_{\text{host}}$ = '+ str(theta_cell)+ r', $\theta_{\text{sym}} = $'+str(theta_sym), fontsize = 15)
plt.tight_layout()
plt.savefig('Figures/'+extract_feature+'_epsilon_host_'+str(epsilon_cell)+'_epsilon_sym_'+str(epsilon_sym)+'_theta_host_'+ str(theta_cell)+ '_theta_sym_'+str(theta_sym)+'.png')
plt.show()

df_plot = df_tmp.pivot(columns = 'sim', index = 'generation')[['theta_cell','theta_sym']]
df_plot.plot()
plt.ylabel(r'$\theta_{\text{host}}$, $\theta_{\text{host}}$', fontsize =16)
plt.title(r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym)+r'$, \theta_{\text{host}}$ = '+ str(theta_cell)+ r', $\theta_{\text{sym}} = $'+str(theta_sym), fontsize = 15)
plt.tight_layout()
plt.savefig('Figures/thetas_zoom_in_epsilon_host_'+str(epsilon_cell)+'_epsilon_sym_'+str(epsilon_sym)+'_theta_host_'+ str(theta_cell)+ '_theta_sym_'+str(theta_sym)+'.png')
plt.show()



