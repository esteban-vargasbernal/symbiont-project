import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random
import pandas as pd
from scipy.stats import norm

from matplotlib.patches import Circle


import sys

from collections import Counter



type_plot = 'I' # only change this line

df = pd.read_csv('Data/df_all_'+type_plot+'.csv')
df_angle = pd.read_csv('Data/df_angle_'+type_plot+'.csv')



scenarios = ['Mutualism', 'Predator-prey', 'Parasitism', 'Competition']
colors = {'Mutualism':"red",'Predator-prey':"blue", 'Parasitism':"green", 'Competition': "black"}


epsilon_cell = 0.001
epsilon_sym = 0.001 #u_sym*L

N_sim = 20
n_split = 6
N_gen_max = 500

e_for_symbionts_ls = np.linspace(-0.5,0.5,n_split)
e_for_cells_ls = np.linspace(-0.5,0.5,n_split)

h_sym = 0
h_cell = 0

e_sym = -0.5
e_cell = -0.5

theta_cell_0 = e_sym + h_cell 
theta_sym_0 = e_cell + h_sym 


for theta_cell_0 in [-0.5,0.5]:
    for theta_sym_0 in [-0.5,0.5]:

        R = 0.01
        circle = Circle((theta_cell_0, theta_sym_0), radius = R, color='blue', fill=False, linewidth=2)


                            
        for j in np.arange(N_sim):

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
        #df_angle_tmp = df_angle_tmp[df_angle_tmp['generation'] < (N_gen_max-1) ]

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
        plt.savefig('Figures/theta_trajectories_one_circle_'+ scenario_start+'_'+type_plot+'.png')
        plt.show()





############################################
############################################

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





############################################
############################################

df_all = df


#scenarios = ['Mutualism', 'Predator-prey', 'Parasitism', 'Competition']
#colors = {'Mutualism':"red",'Predator-prey':"blue", 'Parasitism':"green", 'Competition': "black"}




extract_feature = 'w_cell' # 'alive_cell', 'alive_sym_in', 'alive_sym_out', 's_cell'
N_sim_max = 5
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



