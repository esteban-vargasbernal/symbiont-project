import numpy as np

import matplotlib.pyplot as plt
import math
import copy
import random
import pandas as pd
from scipy.stats import norm

from matplotlib.patches import Circle

import functions_symbionts as sym

import sys

from collections import Counter




type_plot = 'single_mut'

df_angle = pd.read_csv('Data/df_angle_'+type_plot+'.csv')

scenarios = ['Mutualism', 'Predator-prey', 'Parasitism', 'Competition']
colors = {'Mutualism':"red",'Predator-prey':"blue", 'Parasitism':"green", 'Competition': "black"}


epsilon_cell = 0.001
epsilon_sym = 0.001 #u_sym*L

N_sim = 10
n_split = 12
N_gen_max = 500

e_for_cells_ls = np.linspace(-0.75,0.25,n_split)
e_for_symbionts_ls = np.linspace(-0.75,0.25,n_split)


h_sym = 0.25
h_cell = 0.25

theta_sym_v = e_for_cells_ls + h_sym
theta_cell_v = e_for_symbionts_ls + h_cell

mean_xy_m = np.zeros((n_split,n_split))
mean_generation_m = np.zeros((n_split,n_split))
p_value_x_m = np.zeros((n_split,n_split))
p_value_y_m = np.zeros((n_split,n_split))
mean_x_m = np.zeros((n_split,n_split))
mean_y_m = np.zeros((n_split,n_split))

R = 0.01

for i in np.arange(n_split):
    for j in np.arange(n_split):

        e_sym = e_for_symbionts_ls[i]  
        e_cell = e_for_cells_ls[j]

        theta_cell_0 = e_sym + h_cell 
        theta_sym_0 = e_cell + h_sym 


        
        df_angle_tmp = df_angle[np.round(df_angle['theta_sym'],2) == np.round(theta_sym_0,2)]
        df_angle_tmp = df_angle_tmp[np.round(df_angle_tmp['theta_cell'],2)==np.round(theta_cell_0,2)]
        #print(df_angle_tmp.shape)
        #df_angle_tmp = df_angle_tmp[df_angle_tmp['generation'] < (N_gen_max-1) ]

        angles_v = df_angle_tmp['angle']
        generation_v = df_angle_tmp['generation']
        #print(angles_v)

        mean_x = np.mean( np.cos(angles_v))
        mean_y = np.mean( np.sin(angles_v))
        mean_xy = np.sqrt(mean_x**2 + mean_y**2)
        #print(mean_angle)
        sd_x = np.std(np.cos(angles_v))
        sd_y = np.std(np.sin(angles_v))
        sd_xy = np.sqrt(sd_x**2 + sd_y**2)
        
        z_score_x = mean_x/sd_x
        z_score_y = mean_y/sd_y

        p_value_x = 1 - norm.cdf(z_score_x)
        p_value_y = 1 - norm.cdf(z_score_y)

        mean_generation = np.mean(generation_v)/N_gen_max
        
        mean_xy_m[i,j] = mean_xy
        mean_generation_m[i,j] = mean_generation
        p_value_x_m[i,j] = p_value_x
        p_value_y_m[i,j] = p_value_y

        plt.grid()
        plt.arrow(theta_cell_0, theta_sym_0, 10*R/(1+sd_xy**2)*mean_x, 10*R/(1+sd_xy**2)*mean_y, width=0.005)

plt.hlines(y = 0, xmin=-0.55, xmax =0.55, color = "red", linewidth = 3)
plt.vlines(x = 0, ymin =-0.55, ymax = 0.55, color = "red", linewidth = 3)
#plt.xlim(-1,2)
#plt.ylim(-1,1.2)
plt.xlabel(r'$\theta_{\text{host}} = e_{\text{sym}} + h_{\text{host}}$', fontsize = 16)
plt.ylabel(r'$\theta_{\text{sym}} = e_{\text{host}} + h_{\text{sym}}$', fontsize = 16)
plt.title(r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym), fontsize = 18)
#plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, title = 'Scenario of the \n initial condition')
plt.tight_layout()
#plt.savefig('Figures/all_arrows.png')
plt.savefig('Figures/all_arrows_'+type_plot+'.png')
plt.show()

x , y = np.meshgrid(theta_cell_v, theta_sym_v)

plt.pcolormesh(y,x,mean_xy_m, cmap= 'PuOr')
plt.colorbar()
plt.xlabel(r'$\theta_{\text{host}} = e_{\text{sym}} + h_{\text{host}}$', fontsize = 16)
plt.ylabel(r'$\theta_{\text{sym}} = e_{\text{host}} + h_{\text{sym}}$', fontsize = 16)
plt.title(r'$\|(\Delta \theta_{\text{host}}, \Delta \theta_{\text{sym}})\|$, '+r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym), fontsize = 15)
plt.hlines(y = 0, xmin=-0.55, xmax =0.55, color = "red", linewidth = 3)
plt.vlines(x = 0, ymin =-0.55, ymax = 0.55, color = "red", linewidth = 3)
#plt.savefig('Figures/diagonal_change.png')
plt.savefig('Figures/diagonal_change_'+type_plot+'.png')
plt.show()


plt.pcolormesh(y,x,p_value_x_m, cmap= 'PuOr', vmin=0, vmax=1)
plt.colorbar()
plt.xlabel(r'$\theta_{\text{host}} = e_{\text{sym}} + h_{\text{host}}$', fontsize = 16)
plt.ylabel(r'$\theta_{\text{sym}} = e_{\text{host}} + h_{\text{sym}}$', fontsize = 16)
plt.title('p-value for '+ r'$\Delta \theta_{\text{host}}$, ' + r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym), fontsize = 15)
plt.hlines(y = 0, xmin=-0.55, xmax =0.55, color = "red", linewidth = 3)
plt.vlines(x = 0, ymin =-0.55, ymax = 0.55, color = "red", linewidth = 3)
#plt.savefig('Figures/all_x_p_value.png')
plt.savefig('Figures/all_x_p_value_'+type_plot+'.png')
plt.show()

plt.pcolormesh(y,x,p_value_y_m, cmap= 'PuOr', vmin=0, vmax=1)
plt.colorbar()
plt.xlabel(r'$\theta_{\text{host}} = e_{\text{sym}} + h_{\text{host}}$', fontsize = 16)
plt.ylabel(r'$\theta_{\text{sym}} = e_{\text{host}} + h_{\text{sym}}$', fontsize = 16)
plt.title('p-value for '+ r'$\Delta \theta_{\text{sym}}, $' + r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym), fontsize = 15)
plt.hlines(y = 0, xmin=-0.55, xmax =0.55, color = "red", linewidth = 3)
plt.vlines(x = 0, ymin =-0.55, ymax = 0.55, color = "red", linewidth = 3)
#plt.savefig('Figures/all_y_p_value.png')
plt.savefig('Figures/all_y_p_value_'+type_plot+'.png')
plt.show()

plt.pcolormesh(y,x,mean_generation_m, cmap= 'PuOr', vmin=0, vmax=1)
plt.colorbar()
plt.xlabel(r'$\theta_{\text{host}} = e_{\text{sym}} + h_{\text{host}}$', fontsize = 16)
plt.ylabel(r'$\theta_{\text{sym}} = e_{\text{host}} + h_{\text{sym}}$', fontsize = 16)
plt.title('hitting time to the circle '+r'$\epsilon_{\text{host}}$ = '+ str(epsilon_cell)+ r', $\epsilon_{\text{sym}} = $'+str(epsilon_sym), fontsize = 15)
plt.hlines(y = 0, xmin=-0.55, xmax =0.55, color = "red", linewidth = 3)
plt.vlines(x = 0, ymin =-0.55, ymax = 0.55, color = "red", linewidth = 3)
#plt.savefig('Figures/all_generations.png')
plt.savefig('Figures/all_generations_'+type_plot+'.png')
plt.show()