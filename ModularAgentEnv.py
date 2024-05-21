#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The following code builds on the use of Modular agents proposed by 
Dulberg et al (2023) and applies it to the homeostatic reinforcement 
learning theory developed by Keramati and Gutkin (2014). It allows for the 
simulation of the behavior of a modular agent composed of two sub-agents,
responsible for maintaining the drug-related and hydration variables 
respectively close to their HS. """


import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import plotly.graph_objects as go
import matplotlib.animation as animation
import matplotlib.patches as mpatches
#np.random.seed(12)

###############################################################################

# Define the size of the environment
maxh = 8
#np.random.seed(12)

class Modules:
    def __init__(self, n, m, learning_rate, discount_factor, epsilon):
        self.n = n
        self.m = m
        self.state_space_drug = [i for i in range(-maxh, maxh + 1)] 
        self.state_space_physio = [i for i in range(-maxh, maxh + 1)]
        self.action_space = [0, 1, 2, 3]  # 0: eat sugar, 1: drink water, 2: not drink, 3: not drug
        self.hstardrug = 8 
        self.hstarphysio = 0
        self.current_state_drug = np.random.choice(range(-maxh, maxh + 1)) 
        self.current_state_physio = np.random.choice(range(-maxh, maxh + 1)) 
        self.max_hydration = maxh
        self.max_drug = maxh
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Initialize Q-tables of both modules with zeros
        self.q_table_drug = np.zeros((len(self.state_space_drug), len(self.action_space)))
        self.q_table_physio = np.zeros((len(self.state_space_physio), len(self.action_space)))        
        
    def calculate_drive_1D(self, state, hstar):
        "Computes the one-dimensional drive of a 1D state"
        return np.power(np.abs(hstar - state)**self.n, 1/self.m)
    
    def calculate_drive(self, state):
        "Computes the 2-dimensional drive of a 2D state"
        return np.power(np.abs((self.hstardrug - state[0]))**self.n + np.abs((self.hstarphysio - state[1]))**self.n, 1/self.m)

        
    def eps_greedy(self, state, state_space,q_table):
        "Returns an action in the action space using epsilon-greedy policy"
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            state_index = state_space.index(state)
            return np.argmax(q_table[state_index, :])
        
        
    def update_q_table(self, state, state_space, action, reward, next_state, q_table):
        "Updates the Q-table of a module"
        current_index = state_space.index(state)
        next_index = state_space.index(next_state)

        # Q-learning update rule
        q_table[current_index, action] += self.learning_rate * (reward + self.discount_factor * np.max(q_table[next_index, :]) - q_table[current_index, action])


    def Modules_step(self):
        """Executes a step : an action is selected, the next state, drive and
        reward is computed by each module, and their respective Q-tables are 
        updated. The function returns the state of the drug and hydration 
        sub-agents, the reward received, and whether the episode is over 
        after this step """
        
        # Choose action using epsilon-greedy action selection
        action_drug = self.eps_greedy(self.current_state_drug, self.state_space_drug,self.q_table_drug)
        action_physio = self.eps_greedy(self.current_state_physio, self.state_space_physio, self.q_table_physio)
        
        ### STEP FOR DRUG MODULE ###
        # Compute the 1D drive
        current_drive = self.calculate_drive_1D(self.current_state_drug, self.hstardrug)

        # Calculate the next state after taking the action for drug sub-agent
        if action_drug == 0:  #Take drug
            next_state_drug = min(self.current_state_drug + 1, self.max_drug)
        elif action_drug == 1:  #Drink water
            next_state_drug = self.current_state_drug
        elif action_drug == 2:  #Not drug
            next_state_drug = max(self.current_state_drug - 1, -self.max_drug)
        else:  #Not drink
            next_state_drug = self.current_state_drug
        
        # Calculate the drive of the sub-agent's next state
        next_drive = self.calculate_drive_1D(next_state_drug, self.hstardrug)
        
        # Calculate the reward of the state transition
        reward = current_drive - next_drive
        
        # Update the Q-table of the drug sub-agent
        self.update_q_table(self.current_state_drug,self.state_space_drug, action_drug, reward, next_state_drug, self.q_table_drug)
        
        
        ### STEP FOR HYDRATION MODULE ###
        # Compute the 1D drive
        current_drive = self.calculate_drive_1D(self.current_state_physio, self.hstarphysio)
        
        # Calculate the next state after taking the action for hydration sub-agent
        if action_physio == 0:  # Take drug
            next_state_physio = self.current_state_physio
        elif action_physio == 1:  # Drink water
            next_state_physio = min(self.current_state_physio + 1, self.max_hydration)
        elif action_physio == 2:  # Not drug
            next_state_physio = self.current_state_physio
        else:  # Not drink
            next_state_physio = max(self.current_state_physio - 1, -self.max_hydration)
            
        # Calculate the drive of the sub-agent's next state
        next_drive = self.calculate_drive_1D(next_state_physio, self.hstarphysio)
        
        # Calculate the reward of the state transition
        reward = current_drive - next_drive
        
        # Update the Q-table of the hydration sub-agent
        self.update_q_table(self.current_state_physio,self.state_space_physio, action_physio, reward, next_state_physio, self.q_table_physio)
        
        # Update the states of sub-agents
        self.current_state_drug = next_state_drug
        self.current_state_physio = next_state_physio

        # Check if the episode is done
        if self.current_state_drug == self.hstardrug and self.current_state_physio == self.hstarphysio : 
            done = True # Episode is done if both sub-agents reached their HS
        else :
            done = False

        return self.current_state_drug, self.current_state_physio, reward, done
    
    def Modules_reset(self):
        "Resets variables after the end of an episode"
        self.current_state_drug = np.random.choice(range(-maxh, maxh+1))
        self.current_state_physio = np.random.choice(range(-maxh, maxh+1))
        
        return self.current_state_drug, self.current_state_physio

    def visualize_qtables(self):
        "Returns a figure containing the Q-tables of the two modules"

        fig, axs = plt.subplots(1, 2, figsize=(7, 10))
        actions = ['take drug', 'drink water', 'not take drug', 'not drink']
        
        # Plot for Drug Module
        im1 = axs[0].imshow(self.q_table_drug, cmap='Spectral')  
        axs[0].set_title('Drug Module')
        axs[0].set_yticks(np.arange(self.q_table_drug.shape[0]), np.arange(-maxh, maxh+1))
        axs[0].set_xticks(np.arange(self.q_table_drug.shape[1]), actions, rotation=90)
        fig.colorbar(im1, ax=axs[0])  
        
        # Plot for Hydration Module
        im2 = axs[1].imshow(self.q_table_physio, cmap='Spectral')  
        axs[1].set_title('Hydration Module')
        axs[1].set_yticks(np.arange(self.q_table_physio.shape[0]), np.arange(-maxh, maxh+1))
        axs[1].set_xticks(np.arange(self.q_table_physio.shape[1]), actions, rotation=90)
        fig.colorbar(im2, ax=axs[1])  
        
        fig.suptitle(f'n = {self.n}, m = {self.m} ', fontsize=20)
        plt.show()
        
    def get_q_tables(self):
        "Returns the Q-tables of the two modules"
        return self.q_table_drug, self.q_table_physio
    
    def Q_Greatest_Mass(self):
        """Returns the Q-table with optimal actions as computed with the 
        Greatest-Mass Q-Learning action selection mechanism. Returns the Q-table
        and the table of optimal actions."""
        
        D = self.q_table_drug
        P = self.q_table_physio
        Q = np.zeros((D.shape[0], P.shape[0]))
        O_action_matrix = np.zeros((D.shape[0], P.shape[0]), dtype=int)
        summed_actions = np.zeros(4)
        for i in range(D.shape[0]):
            for j in range(P.shape[0]):
                for k in range(4):
                    summed_actions[k] = D[i, k] + P[j, k]
                Q[i,j] = max(summed_actions)
                O_action_matrix[i,j] = np.argmax(summed_actions)
        
        return Q,O_action_matrix
    
    
    def Top_Q(self):
        """Returns the Q-table with optimal actions as computed with the 
        Top-Q-Learning action selection mechanism. Returns the Q-table
        and the table of optimal actions."""
        
        D = self.q_table_drug
        P = self.q_table_physio
        Q = np.zeros((D.shape[0], P.shape[0]))
        O_actions = np.zeros((D.shape[0], P.shape[0]), dtype=int)
        for i in range(D.shape[0]):
            for j in range(P.shape[0]):
                max_D = np.max(D[i, :])
                max_P = np.max(P[j, :])
                max_tot = max(max_D, max_P)
                
                if max_D >= max_P:
                    max_action = np.argmax(D[i, :])
                else:
                    max_action = np.argmax(P[j, :])
                
                Q[i, j] = max_tot
                O_actions[i, j] = max_action
        
        return Q, O_actions
        
    def calculate_optimal_action_rewards(self):
        "Computes the rewards received when taking the optimal action"
        
        optimal_action_rewards = np.zeros((2*maxh + 1, 2*maxh + 1))
        
        # To use the Top-Q-Learning action selection uncomment below
        #Q, optimal_actions = self.Top_Q()
        
        # To use the Greatest-Mass action selection uncomment below
        Q, optimal_actions = self.Q_Greatest_Mass()
        
        for i in range(2*maxh + 1):
            for j in range(2*maxh + 1):
                state = (i - maxh, j - maxh)  
                optimal_action = optimal_actions[i, j]
                if optimal_action == 0:  # Take drug
                    next_state = (min(state[0] + 1, self.max_drug), state[1])
                elif optimal_action == 1:  # Drink water
                    next_state = (state[0], min(state[1] + 1, self.max_hydration))
                elif optimal_action == 2:  # Not drug
                    next_state = (max(state[0] - 1, -self.max_drug), state[1])
                else:  # Not drink
                    next_state = (state[0], max(state[1] - 1, -self.max_hydration))
                current_drive = self.calculate_drive(state)
                next_drive = self.calculate_drive(next_state)
                reward = current_drive - next_drive
                optimal_action_rewards[i, j] = reward
                
        return optimal_action_rewards
        
    def visualize_topographic_rewards(self):
        "Returns a 3D plot of the rewards obtained by taking a step toward the HS"
        optimal_action_rewards = self.calculate_optimal_action_rewards()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        X = np.arange(-maxh, maxh + 1)
        Y = np.arange(-maxh, maxh + 1)
        X, Y = np.meshgrid(X, Y)
        Z = optimal_action_rewards
        
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f'Topographic map of rewards of optimal action for n = {self.n}, m = {self.m}' )
        ax.set_xlabel('Hydration Variable')
        ax.set_ylabel('Drug-related Variable')
        ax.set_zlabel('Reward')
        plt.show()
        
        
    def visualize_state_value_and_policy(self):
        "Returns a plot of the Q-map and optimal policy"
        
        # To use the Top-Q-Learning action selection uncomment below
        #state_values, optimal_actions = self.Top_Q()
        
        # To use the Greatest-Mass action selection uncomment below
        state_values, optimal_actions = self.Q_Greatest_Mass()
        
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Plot state-values
        im = ax.imshow(state_values, cmap='Spectral', origin='lower')
    
        # Arrows for optimal actions
        for i in range(2*maxh + 1):
            for j in range(2*maxh + 1):
                action = optimal_actions[i, j]
                if action == 0:  # Take drug
                    ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.05, fc='black', ec='black')
                elif action == 1:  # Drink water
                    ax.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
                elif action == 2:  # Not drug
                    ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.05, fc='black', ec='black')
                elif action == 3:  # Not drink
                    ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
        

        plt.xlabel('Hydration Variable ($h_p$)', fontsize=25)
        plt.ylabel('Drug-related Variable ($h_d$)', fontsize=25)
        plt.title('State-Action-Value and Optimal Policy Map')
        plt.xticks(range(2*maxh + 1),np.arange(-maxh, maxh + 1),fontsize = 15)
        plt.yticks(range(2*maxh + 1),np.arange(-maxh, maxh + 1),fontsize = 15)
        colorbar = plt.colorbar(im,label='Q-value')
        cax = colorbar.ax
        cax.tick_params(axis='y', labelsize=14)
        colorbar.set_label('Q-value', fontsize=25)
        plt.grid(False)
        plt.suptitle(f'n = {self.n}, m = {self.m}', fontsize=23)
        plt.show()
                
        

###############################################################################

# Define the environment       
env1 = Modules(n= 2, m= 2, learning_rate= 0.4, discount_factor = 0.5, epsilon = 0.2)

# Run the simulation
num_episodes = 10000

for episode in range(num_episodes):
    drug_state, physio_state = env1.Modules_reset()
    
    while True:
        # Take an action using Q-learning and epsilon-greedy rule
        drug_state, physio_state, reward, done = env1.Modules_step()
            
        if done:
            break # Episode ends when both sub-agents have reached their HS

# Display the resulting Q-table with optimal actions
env1.visualize_state_value_and_policy()



