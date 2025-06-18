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



#df_all = pd.read_csv('Data/df_N_1000.csv')

df_all = pd.read_csv('Data/df_small_N_1000_3.csv')


#scenarios = ['Mutualism', 'Predator-prey', 'Parasitism', 'Competition']
#colors = {'Mutualism':"red",'Predator-prey':"blue", 'Parasitism':"green", 'Competition': "black"}


epsilon_cell = 0.001
epsilon_sym = 0.001 #u_sym*L

n_split = 4

e_list = np.linspace(-0.75,0.25,n_split)
e_cell = e_list[1]
e_sym = e_list[1]


extract_feature = 'alive_sym_in' # 'alive_cell', 'alive_sym_in', 'alive_sym_out', 's_cell'


N_sim_max = 3


df = df_all[ np.logical_and(df_all['epsilon_cell']==epsilon_cell, df_all['epsilon_sym']==epsilon_sym)]
df_tmp = df[np.round(df['e_sym_0'],2) == np.round(e_sym,2)]
df_tmp = df_tmp[np.round(df_tmp['e_cell_0'],2)==np.round(e_cell,2)]
df_tmp = df_tmp[df_tmp['sim']<N_sim_max]

theta_cell = np.round(df_tmp["theta_cell"].iloc[0],2)
theta_sym = np.round(df_tmp['theta_sym'].iloc[0],2)

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



