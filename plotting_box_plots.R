setwd("/Users/estebanvargasbernal/Documents/github/symbiont-project")

library(arrangements)
library(matrixcalc)
library(ggplot2)
library(parallel)
ncores <- detectCores() - 1
RNGkind("L'Ecuyer-CMRG")
library(igraph)
library(latex2exp)
library(sfsmisc) 
library('plot.matrix')
library(nnls)
library(tidyverse)
library(dplyr)

df_sum <- read.csv(file = "./Data/df_sum.csv", header = TRUE,  sep = ",", row.names = NULL,  stringsAsFactors = FALSE)
df_sum <-  arrange(df_sum,desc(theta_sym))


n_split <- 6
epsilon_cell_0 <- 0.001
epsilon_sym_0 <- 0.001


df_sum_tmp <- df_sum %>% mutate(theta_sym = as.character(round(theta_sym,2)), theta_cell = as.character(round(theta_cell,2))) %>%  filter(epsilon_cell == epsilon_cell_0, epsilon_sym == epsilon_sym_0)  %>%
transform(theta_sym = factor(theta_sym, levels = as.character(unique(round(df_sum$theta_sym,2)))),
theta_cell = factor(theta_cell, levels = as.character(unique(round(df_sum$theta_cell,2)))),
scenario_end = factor(scenario_end, levels = c("Mutualism",  'Predator-prey', 'Parasitism', 'Competition')))

head(df_sum_tmp)

plot_cor <-  df_sum_tmp %>%
  group_by(theta_sym, theta_cell) %>%   
  ggplot( aes(x = scenario_end)) +
  geom_bar(aes(y = ..prop.., group = 1, fill = scenario_end), stat =  'count') +
  scale_fill_manual(values = c( "red", "blue","green","black" )) +
  facet_wrap(~ theta_sym + theta_cell, labeller = labeller(
            theta_cell = ~ paste("Theta host: ", .),
            theta_sym = ~ paste("Theta_sym: ", .) ,nrow = n_split)) +
  labs(x = 'Theta host', y = 'Theta sym', title = paste0('Distributions of final scenario over different simulations')) +
  labs(fill = "Scenario")+
  theme(axis.text = element_text(size = 15), plot.title = element_text(size = 14),
        axis.title.x = element_text(size = 16), axis.title.y = element_text(size = 16),
        legend.text = element_text(size = 10), axis.text.x=element_blank(), axis.ticks.x = element_blank())

plot_cor

ggsave(plot_cor, file = paste0('Figures/distribution_final_senario_epsilon_cell_',epsilon_cell_0,'_epislon_sym_',epsilon_sym_0,'.eps'))

