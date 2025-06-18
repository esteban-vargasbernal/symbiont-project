#setwd("/Users/estebanvargasbernal/Documents/github/symbiont-project")

library(arrangements)
library(matrixcalc)
library(ggplot2) 
library(parallel)
ncores <- detectCores() - 1
RNGkind("L'Ecuyer-CMRG")
library(latex2exp)
library(sfsmisc)
library(nnls)
library(tidyverse)
library(dplyr)


df <-read.csv(file = 'Data/df_small_N_1000_3.csv', header = TRUE,  sep = ",")

n_split <- 4


df_tmp <- df  %>%
group_by(e_cell_0, e_sym_0, sim) %>%
mutate(theta_sym = first(theta_sym), theta_cell = first(theta_cell))
    
df_tmp <- df_tmp %>%
mutate(theta_sym = as.character(round(theta_sym,2)), 
theta_cell = as.character(round(theta_cell,2))) 

df_tmp <- df_tmp %>%
    transform(theta_sym = factor(theta_sym, levels = c("0.5","0.17","-0.17","-0.5") ),
    theta_cell = factor(theta_cell, levels = c("-0.5","-0.17","0.17","0.5") ),
    scenario = factor(scenario, levels = c("Mutualism",  'Predator-prey', 'Parasitism', 'Competition')))

### Plot duration in each sscenario

for(epsilon_cell_0 in c(0.001, 0.00001)){
    for(epsilon_sym_0 in c(0.001, 0.00001)){

        df_plot <- df_tmp  %>% filter(epsilon_cell == epsilon_cell_0, epsilon_sym == epsilon_sym_0) 

        plot_cor <-  df_plot %>%
        group_by(e_cell_0, e_sym_0, sim) %>% 
        ggplot( aes(x = scenario)) +
        geom_bar(aes( fill = scenario), stat = 'count') + 
        scale_fill_manual(values = c( "red", "blue","green","black" )) +
        facet_wrap(~ theta_sym + theta_cell, labeller = labeller(
                    theta_cell = ~ paste("Theta host: ", .),
                    theta_sym = ~ paste("Theta_sym: ", .) ,nrow = n_split)) +
        labs(x = 'Scenario', y = 'Number of generations in each scenario', title = paste0('Distributions of time spent in each scenario over different simulations for 10000 generations, \n epsilon_host = ', epsilon_cell_0,', epislon_sym = ',epsilon_sym_0, ', 3 simulations for each parameter combination')) +
        labs(fill = "Scenario")+
        theme(axis.text = element_text(size = 15), plot.title = element_text(size = 14),
                axis.title.x = element_text(size = 16), axis.title.y = element_text(size = 16), # nolint
                legend.text = element_text(size = 15), axis.text.x=element_blank(), axis.ticks.x = element_blank())

        #plot_cor
        ggsave(plot_cor, file = paste0('Figures/distribution_duration_each_scenario_epsilon_cell_',epsilon_cell_0,'_epislon_sym_',epsilon_sym_0,'.pdf'))
    }
}

df_tmp_2 <- df_tmp %>% group_by(theta_cell,theta_sym,sim, epsilon_cell, epsilon_sym) %>%
              summarise(scenario_end_short = factor(first(scenario_end), levels = c("Mutualism",  'Predator-prey', 'Parasitism', 'Competition') ))

### Plot final scenario 

for(epsilon_cell_0 in c(0.001, 0.00001)){
    for(epsilon_sym_0 in c(0.001, 0.00001)){

        df_plot_2 <- df_tmp_2  %>% filter(epsilon_cell == epsilon_cell_0, epsilon_sym == epsilon_sym_0) 

        plot_cor <-  df_plot_2 %>%
        group_by(theta_cell, theta_sym, sim) %>% 
        ggplot( aes(x = scenario_end_short)) +
        #geom_bar(aes( fill = scenario), position = 'fill') + 
        geom_bar(aes( fill = scenario_end_short), stat = 'count') + 
            #geom_bar(aes(y = ..prop.., group = 1, fill = scenario), stat =  'count') +
        scale_fill_manual(values = c( "red", "blue","green","black" )) +
        facet_wrap(~ theta_sym + theta_cell, labeller = labeller(
                    theta_cell = ~ paste("Theta host: ", .),
                    theta_sym = ~ paste("Theta_sym: ", .) ,nrow = n_split)) +
        labs(x = 'Scenario', y = 'Number of generations in each scenario', title = paste0('Distributions of final scenario over different simulations for 10000 generations, \n epsilon_host = ', epsilon_cell_0,', epislon_sym = ',epsilon_sym_0, ', 3 simulations for each parameter combination')) +
        labs(fill = "Scenario")+
        theme(axis.text = element_text(size = 15), plot.title = element_text(size = 14),
                axis.title.x = element_text(size = 16), axis.title.y = element_text(size = 16), # nolint
                legend.text = element_text(size = 15), axis.text.x=element_blank(), axis.ticks.x = element_blank())

        ggsave(plot_cor, file = paste0('Figures/final_scenario_epsilon_cell_',epsilon_cell_0,'_epislon_sym_',epsilon_sym_0,'.pdf'))
    }
}

