#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The following code builds on the homeostatic reinforcement learning theory 
 developed by Keramati and Gutkin (2014).It allows for the simulation of an 
 healthy agent in a homeostatic state space and can be modulated for the agent 
 to display compulsive drug seeking. Two variants of the mathematical 
 expression of the drive can be selected. """


import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import plotly.graph_objects as go
#np.random.seed(12)

###############################################################################

# Define the size of the environment
maxh = 8

class HomeoEnv2D:
    def __init__(self, n, m, lam, p, learning_rate, discount_factor, epsilon):
        self.n = n
        self.m = m
        self.lam = lam # set to [1,1] if not using the elliptic drive
        self.p = p # set to self.n if not using the p-drive
        self.state_space = [(i, j) for i in range(-maxh, maxh + 1) for j in range(-maxh, maxh + 1)]
        self.action_space = [0, 1, 2, 3, 4]  # 0: eat sugar, 1: drink water, 2: not drink, 3: not drug, 4 : do nothing
        self.hstar = (0, 0)
        self.current_state = tuple(np.random.choice(range(-maxh, maxh+1), size=2))  
        self.max_hydration = maxh
        self.max_drug = maxh
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        
        self.q_table = np.zeros((len(self.state_space), len(self.action_space))) # Initialize Q-table with zeros
        self.v_table = np.zeros(len(self.state_space)) # Initialize V-table with zeros

    def calculate_drive(self, state):
        "Computes the drive of a state"
        return np.power( self.lam[0]*np.abs((self.hstar[0] -state[0]))**self.n + self.lam[1]*np.abs((self.hstar[1] - state[1]))**self.p, 1/self.m)

    def eps_greedy(self, state):
        "Returns an action in the action space using epsilon-greedy policy"
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            state_index = self.state_space.index(state)
            return np.argmax(self.q_table[state_index, :])

    def update_q_table(self, state, action, reward, next_state):
        "Updates the Q-table"
        current_index = self.state_space.index(state)
        next_index = self.state_space.index(next_state)

        # Q-learning update rule
        self.q_table[current_index, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_index, :]) - self.q_table[current_index, action])
        
    def update_v_table(self, state, reward, next_state):
        "Updates the V-table"
        current_index = self.state_space.index(state)
        next_index = self.state_space.index(next_state)
        
        self.v_table[current_index] += self.learning_rate*(reward + self.discount_factor*self.v_table[next_index] - self.v_table[current_index])
        
            
    def step(self):
        """Executes a step : returns the state of the agent, the HS, the reward it received, 
        whether the episode is over after this step and the action that was taken """
        
        # Choose action using epsilon-greedy rule
        action = self.eps_greedy(self.current_state)

        # Calculate the drive of the current state
        current_drive = self.calculate_drive(self.current_state)

        # Calculate the next state after taking the action
        if action == 0:  # Take drug
            next_state = (min(self.current_state[0] + 1, self.max_drug), self.current_state[1])
            
            # uncomment below for an allostatic increase of the drug-related HS
            #newhstar = (min(self.hstar[0] + 1,maxh), self.hstar[1])
            #self.hstar = newhstar
            
            # uncomment below for reduction of p 
            #self.p = round(max(0.01, self.p - 0.01), 3)
            
            #uncomment below for reduction of lambda_w
            #self.lam = [self.lam[0],round(max(0.01, self.lam[1] - 0.01), 3)]
            
        elif action == 1:  # Drink water
            next_state = (self.current_state[0], min(self.current_state[1] + 1, self.max_hydration))
        elif action == 2:  # Not drug
            next_state = (max(self.current_state[0] - 1, -self.max_drug), self.current_state[1])
        elif action == 3:  # Not drink
            next_state = (self.current_state[0], max(self.current_state[1] - 1, -self.max_hydration))
        else :  #Do nothing
            next_state = (self.current_state[0], self.current_state[1])
        
        # Calculate the drive of the next state
        next_drive = self.calculate_drive(next_state)
        
        # Compute the reward
        reward = current_drive - next_drive

        # Update Q-table
        self.update_q_table(self.current_state, action, reward, next_state)
        
        # Update V-table
        self.update_v_table(self.current_state, reward, next_state)

        # Update the state
        self.current_state = next_state
       
        # Check if the episode is done
        done = self.current_state == self.hstar # Episode ends when the agent reaches the HS

        return self.current_state, self.hstar, reward, done, action

    def reset(self):
        "Resets variables after the end of an episode"
        
        self.current_state = tuple(np.random.choice(range(-maxh, maxh+1), size=2)) # Agent is spawned at a random state
        #self.hstar = (0,0) # uncomment for an allostatic increase of the drug-related HS with reset
        return self.current_state, self.hstar
    
        
    def visualize_vtable(self):
        "Returns a plot of the Value table"
        
        vtable_values = self.v_table.reshape((2 * maxh + 1, 2 * maxh + 1))
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(vtable_values, cmap='Spectral', origin='lower')
        plt.title(f'State value map for n = {self.n}, m = {self.m}')
        plt.colorbar(im, label='state value')
        fig.text(0.02, 0.95, f'alpha = {self.learning_rate}, gamma = {self.discount_factor}, epsilon = {self.epsilon}', fontsize=12, ha='left', va='center')
        plt.xlabel('Hydration Variable')
        plt.ylabel('Drug-related Variable')
        plt.xticks(range(2*maxh + 1),np.arange(-maxh, maxh + 1) )
        plt.yticks(range(2*maxh + 1),np.arange(-maxh, maxh + 1))
        plt.show()
    
    
    def visualize_q_values(self):
        "Returns a figure containing the Q-table for each action"
        
        drug_values = self.q_table[:, 0]
        drink_water_values = self.q_table[:, 1]
        not_drug_values = self.q_table[:, 2]
        not_water_values = self.q_table[:, 3]
        do_nothing_values = self.q_table[:, 4]
        
        drug_values = drug_values.reshape((2 * maxh + 1, 2 * maxh + 1))
        drink_water_values = drink_water_values.reshape((2 * maxh + 1, 2 * maxh + 1))
        not_drug_values = not_drug_values.reshape((2 * maxh + 1, 2 * maxh + 1))
        not_water_values = not_water_values.reshape((2 * maxh + 1, 2 * maxh + 1))
        do_nothing_values = do_nothing_values.reshape((2 * maxh + 1, 2 * maxh + 1))
        drug_levels = np.arange(-maxh, maxh + 1)
        hydration_levels = np.arange(-maxh, maxh + 1)
        drug_grid, hydration_grid = np.meshgrid(drug_levels, hydration_levels)
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        # Plot for taking drugs
        im = axes[0].imshow(drug_values, cmap='viridis', origin='lower', extent=[-maxh, maxh, -maxh, maxh])
        axes[0].set_title('Take Drug')
        axes[0].set_xlabel('Hydration Variable')
        axes[0].set_ylabel('Drug-related Variable')
        axes[0].set_xticks(np.arange(-maxh, maxh + 1, 5))
        axes[0].set_yticks(np.arange(-maxh, maxh + 1, 5))
        fig.colorbar(im, ax=axes[0])
        
        # Plot for drinking water
        im = axes[1].imshow(drink_water_values, cmap='viridis', origin='lower', extent=[-maxh, maxh, -maxh, maxh])
        axes[1].set_title('Drink Water')
        axes[1].set_xlabel('Hydration Variable')
        axes[1].set_ylabel('Drug-related Variable')
        axes[1].set_xticks(np.arange(-maxh, maxh + 1, 5))
        axes[1].set_yticks(np.arange(-maxh, maxh + 1, 5))
        fig.colorbar(im, ax=axes[1])
        
        # Plot for not taking drugs
        im = axes[2].imshow(not_drug_values, cmap='viridis', origin='lower', extent=[-maxh, maxh, -maxh, maxh])
        axes[2].set_title('Not Drug')
        axes[2].set_xlabel('Hydration Variable')
        axes[2].set_ylabel('Drug-related Variable')
        axes[2].set_xticks(np.arange(-maxh, maxh + 1, 5))
        axes[2].set_yticks(np.arange(-maxh, maxh + 1, 5))
        fig.colorbar(im, ax=axes[2])
        
        # Plot for not drinking water
        im = axes[3].imshow(not_water_values, cmap='viridis', origin='lower', extent=[-maxh, maxh, -maxh, maxh])
        axes[3].set_title('Not Water')
        axes[3].set_xlabel('Hydration Variable')
        axes[3].set_ylabel('Drug-related Variable')
        axes[3].set_xticks(np.arange(-maxh, maxh + 1, 5))
        axes[3].set_yticks(np.arange(-maxh, maxh + 1, 5))
        fig.colorbar(im, ax=axes[3])
        
        # Plot for doing nothing
        im = axes[4].imshow(do_nothing_values, cmap='viridis', origin='lower', extent=[-maxh, maxh, -maxh, maxh])
        axes[4].set_title('Do nothing')
        axes[4].set_xlabel('Hydration Variable')
        axes[4].set_ylabel('Drug-related Variable')
        axes[4].set_xticks(np.arange(-maxh, maxh + 1, 5))
        axes[4].set_yticks(np.arange(-maxh, maxh + 1, 5))
        fig.colorbar(im, ax=axes[4])
        
        plt.suptitle(f'n = {self.n}, m = {self.m}', fontsize=23)
        
        plt.tight_layout()
        plt.savefig('/Users/clementineguillemet/Desktop/ensmaster/stage/S3/code/figures/Newfile/donothing.png')
        plt.show()
        
    def visualize_state_value_and_policy(self):
        "Returns a plot of the Q-map and optimal policy"
        
        state_values = np.max(self.q_table, axis=1).reshape((2*maxh + 1, 2*maxh + 1))
        optimal_actions = np.argmax(self.q_table, axis=1).reshape((2*maxh + 1, 2*maxh + 1))
        
        fig, ax = plt.subplots(figsize=(12, 9))
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
                else:
                    ax.scatter(j,i, marker = 'x', color = 'black', s = 200)
        
        # Plot the optimal trajectory
        i,j=(2,2)
        trajectory_xs = [j]
        trajectory_ys = [i]
        while (i, j) != (8, 8):
            action = optimal_actions[i,j]
            if action == 0:  # up
                i += 1
            elif action == 1:  # right
                j += 1
            elif action == 2:  # down
                i -= 1
            elif action == 3:  # left
                j -= 1
            elif action == 4: # do nothing
                i,j = i,j
            
            trajectory_xs.append(j)
            trajectory_ys.append(i)
        
        ax.plot(trajectory_xs, trajectory_ys, marker='o',markersize = 9, linestyle='-', lw = 3.5, color = 'black')
        
        plt.xlabel('Hydration Variable ($h_w$)', fontsize=25,  fontfamily='serif')
        plt.ylabel('Drug-related Variable ($h_d$)',  fontsize=25, fontfamily='serif')
        plt.title('State-action-value and optimal policy map', fontsize=25,  fontfamily='serif')
        plt.xticks(range(2*maxh + 1),np.arange(-maxh, maxh + 1) ,  fontfamily='serif',fontsize = 15)
        plt.yticks(range(2*maxh + 1),np.arange(-maxh, maxh + 1),  fontfamily='serif',fontsize = 15)
        plt.grid(False)
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            
        })
        colorbar = plt.colorbar(im,label='Q-value')
        cax = colorbar.ax
        cax.tick_params(axis='y', labelsize=14)
        colorbar.set_label('Q-value', fontsize=25)
        # If using normal drive 
        plt.suptitle(f'n = {self.n}, m = {self.m}', fontsize=23,  fontfamily='serif')
        # If using elliptic drive uncomment below
        # plt.suptitle(f'n = {self.n}, m = {self.m}', $\lambda$ = {self.lam} fontsize=23,  fontfamily='serif')
        # If using p-drive uncomment below
        # plt.suptitle(f'n = {self.n}, m = {self.m}', p = {self.p} $\lambda$ = {self.lam} fontsize=23,  fontfamily='serif')

        plt.savefig('/Users/clementineguillemet/Desktop/figuresrap/N1M2control.png')
        plt.show()
        

    def calculate_optimal_action_rewards(self):
        "Computes the rewards received when taking the optimal action"
        
        optimal_action_rewards = np.zeros((2*maxh + 1, 2*maxh + 1))
        optimal_actions = np.argmax(self.q_table, axis=1).reshape((2*maxh + 1, 2*maxh + 1))
        
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
                elif optimal_action == 3: # Not drink
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
        
    def plot_state_visit_freq(self):
        "Returns a plot of the frequency of visit of each state"
        
        total_steps = np.sum(self.visit_count)
        state_visit_freq = np.zeros((2 * maxh + 1, 2 * maxh + 1))
        
        for i in range(2 * maxh + 1):
            for j in range(2 * maxh + 1):
                state = (i - maxh, j - maxh)  
                state_index = self.state_space.index(state)
                state_visit_freq[i, j] = self.visit_count[state_index] / total_steps
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(state_visit_freq, cmap='viridis', origin='lower')
        plt.xlabel('Hydration Variable')
        plt.ylabel('Drug-related Variable')
        plt.title('State Visit Frequency Map')
        plt.xticks(range(2 * maxh + 1), np.arange(-maxh, maxh + 1), labelsize = 12)
        plt.yticks(range(2 * maxh + 1), np.arange(-maxh, maxh + 1),labelsize = 12)
        plt.grid(False)
        plt.colorbar(im, label='Visit Frequency')
        plt.suptitle(f'n = {self.n}, m = {self.m}', fontsize=23)
        plt.savefig('/Users/clementineguillemet/Desktop/ensmaster/stage/S3/code/figures/Newfile/visitfreq.png')
        plt.show()
                    
        
###############################################################################

# Define the environment
env = HomeoEnv2D(n=2, m=2, lam = [1,1], p = 2, learning_rate= 0.4, discount_factor = 0.5, epsilon = 0.4)

# Run the simulation
num_episodes = 10000
for episode in range(num_episodes):
    state,hstar = env.reset()
    
    while True:
        # Take an action using Q-learning and epsilon-greedy
        state, hstar, reward, done, action = env.step()
        
        if done: # The episode ends when the agent reaches the HS
            break

# Display the resulting Q-table with optimal actions
env.visualize_state_value_and_policy()



