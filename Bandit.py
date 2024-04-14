############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

logging.basicConfig
logger = logging.getLogger("MAB Application")

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # log average reward (use f strings to make it informative)
        # log average regret (use f strings to make it informative)
        pass

#--------------------------------------#

class Visualization():
    """
    Class for visualizing results of bandit algorithms.
    """

    def __init__(self):
        """
        Initialize Visualization class.
        """
        pass

    def plot1(self, rewards_egreedy, rewards_thompson):
        """
        Plot cumulative rewards and average reward comparison for both algorithms.

        Parameters:
        - rewards_egreedy (list): List of rewards obtained by Epsilon Greedy algorithm.
        - rewards_thompson (list): List of rewards obtained by Thompson Sampling algorithm.
        """
        # Calculate cumulative rewards
        cumulative_rewards_egreedy = np.cumsum(rewards_egreedy)
        trials_egreedy = np.arange(1, len(rewards_egreedy) + 1)

        cumulative_rewards_thompson = np.cumsum(rewards_thompson)
        trials_thompson = np.arange(1, len(rewards_thompson) + 1)

        # Plot cumulative rewards
        plt.figure(figsize=(10, 5))
        plt.plot(trials_egreedy, cumulative_rewards_egreedy, label="Epsilon Greedy", color='blue')
        plt.plot(trials_thompson, cumulative_rewards_thompson, label="Thompson Sampling", color='orange')
        plt.title("Cumulative Reward Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot average reward on a log scale
        avg_rewards_egreedy = cumulative_rewards_egreedy / trials_egreedy
        avg_rewards_thompson = cumulative_rewards_thompson / trials_thompson

        plt.figure(figsize=(10, 5))
        plt.plot(trials_egreedy, avg_rewards_egreedy, label="Epsilon Greedy", color='blue')
        plt.plot(trials_thompson, avg_rewards_thompson, label="Thompson Sampling", color='orange')
        plt.title("Average Reward Comparison (Log Scale)")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.xscale("log")
        plt.grid(True)
        plt.show()


    def plot2(self, cumulative_rewards_egreedy, cumulative_rewards_thompson, cumulative_regrets_egreedy, cumulative_regrets_thompson):
        """
        Compare E-greedy and Thompson sampling cumulative rewards and regrets.
        """
        plt.plot(cumulative_rewards_egreedy, label='E-greedy')
        plt.plot(cumulative_rewards_thompson, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Reward')
        plt.title('Comparison of Cumulative Rewards')
        plt.legend()
        plt.show()

        plt.plot(cumulative_regrets_egreedy, label='E-greedy')
        plt.plot(cumulative_regrets_thompson, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Regret')
        plt.title('Comparison of Cumulative Regrets')
        plt.legend()
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    An implementation of the Epsilon Greedy algorithm for multi-armed bandit problems.
    Inherits from the Bandit class.
    """
    
    def __init__(self, reward):
        """
        Constructor for the EpsilonGreedy class.
        """
        self.N = 0
        self.reward = reward
        self.reward_estimate = 0
    
    def __repr__(self):
        """
        String representation of the class.
        """
        return f'EpsilonGreedy(Bandit = {self.reward})'
    
    def pull(self):
        """
        Pull the arm of the bandit and generate a random reward.
        """
        return np.random.randn() + self.reward
    
    def update(self, x):
        """
        Updates the reward estimate based on the current reward obtained.
        """
        self.N += 1
        self.reward_estimate = (1 - 1.0 / self.N) * self.reward_estimate + 1.0 / self.N * x

    def experiment(self, bandit_rewards, t, N):
        """
        Run the Epsilon Greedy algorithm on a set of bandits.
        """
        bandits = [EpsilonGreedy(reward) for reward in bandit_rewards]
        
        num_times_explored = 0
        num_times_exploited = 0
        num_optimal = 0
        optimal_j = np.argmax([b.reward for b in bandits])
        print(f'optimal bandit index: {optimal_j}')
        
        # empty array to later add the rewards for inference plots
        eg_rewards = np.empty(N)
        eg_selected_bandit = []
        eps = 1/t

        for i in range(N):

            p = np.random.random() 
            
            # if the random number is smaller than eps we explore a random bandit
            if p < eps:  
                num_times_explored += 1
                j = np.random.choice(len(bandits)) 
            else:
                # if the random number is bigger than eps we explore the bandit with the highest current reward
                num_times_exploited += 1
                j = np.argmax([b.reward_estimate for b in bandits])
            
            # pull the chosen bandit and get the output
            x = bandits[j].pull()
            
            # increases N by 1 and calculates the estimate of the reward
            bandits[j].update(x) 
            
            # if j is the actual optimal bandit, the optimal bandit count increments by 1
            if j == optimal_j:
                num_optimal += 1
            
            # add the selected bandit to the list of selected bandits
            eg_selected_bandit.append(j)
            
            # add the reward to the data
            eg_rewards[i] = x
            
            # decrease the probability of choosing suboptimal (random) bandit (increase t)
            t += 1
            eps = 1 / t
        
        estimated_avg_rewards=[round(b.reward_estimate, 3) for b in bandits]
        print(f'Estimated average reward: epsilon = {eps}: {estimated_avg_rewards}')
        
        all_bandits = pd.DataFrame({"Bandit" : eg_selected_bandit, "Reward" : eg_rewards, "Algorithm" : "Epsilon Greedy"})
        all_bandits.to_csv('Results.csv', mode='a', header=not os.path.exists('Results.csv'), index = False)    
        
        return eg_rewards, num_times_explored, num_times_exploited, num_optimal
    
    def plot_learning_process(self, bandit_rewards, eg_rewards, N):
        """
        Plots the win rate and optimal win rate against the number of trials.
        """        
        cumulative_rewards = np.cumsum(eg_rewards)
        win_rates = cumulative_rewards / (np.arange(N) + 1)
        
        plt.figure(figsize=(10, 8))
        plt.plot(win_rates, label="Win Rate")
        plt.plot(np.ones(N)*np.max(bandit_rewards), label='Optimal Win Rate')
        plt.legend()
        plt.title("Win Rate Convergence Epsilon-Greedy")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.show()

    def report(self, eg_rewards, num_times_explored, num_times_exploited, num_optimal, N):

        print(f"\nTotal Reward Earned: {eg_rewards.sum()}")
        print(f"Average Reward: {np.mean(eg_rewards)}")
        print(f"Overall Win Rate: {eg_rewards.sum() / N :.4f}")
        print(f"# of explored: {num_times_explored}")
        print(f"# of exploited: {num_times_exploited}")
        print(f"# of times selected the optimal bandit: {num_optimal}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    An implementation of the Thompson Sampling algorithm for multi-armed bandit problems.

    Inherits from the Bandit class.
    """
    
    def __init__(self, true_mean):
        """
        Constructor for the ThompsonSampling class.
        """
        self.true_mean = true_mean
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0
        self.sum_x = 0
        
    def __repr__(self):
        """
        String representation of the class.
        """
        return f"A Bandit with {self.true_mean} Win Rate"

    def pull(self):
        """
        Samples a reward from the bandit using its true mean.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        """
        Samples a reward from the bandit using its posterior mean.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        """
        Updates the bandit's posterior mean and precision using the reward received.
        """
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
    
    def experiment(self, bandit_rewards, N):
        """
        Runs the Thompson Sampling algorithm on a set of bandits.
        """
        bandits = [ThompsonSampling(m) for m in bandit_rewards]
        
        sample_points = [100, 1000, 2000, 5000, 10000, 19999]
        
        # empty array to later add the rewards for inference plots
        t_rewards = np.empty(N)
        t_selected_bandit = []
        
        for i in range(N):
            j = np.argmax([b.sample() for b in bandits]) #taking the highest mean position
            
            # make some plots
            if i in sample_points:
                self.plot_bandit_distributions(bandits, i)
            
            # pull the chosen bandit and get the output
            x = bandits[j].pull()

            # increases N by 1, updates lambda and calculates the estimate of the m
            bandits[j].update(x)
            
            # add the reward to the data
            t_rewards[i] = x
            
            # Add the selected bandit to the list
            t_selected_bandit.append(j)
        
        all_bandits = pd.DataFrame({"Bandit" : t_selected_bandit, "Reward" : t_rewards, "Algorithm" : "Thompson Sampling"})
        all_bandits.to_csv('Results.csv', mode='a', header=not os.path.exists('Results.csv'), index = False) 

        return bandits, t_rewards
    
    def plot_learning_process(self, bandit_rewards, t_rewards, N):
        """
        Plots the win rate and optimal win rate against the number of trials.
        """
        cumulative_rewards = np.cumsum(t_rewards)
        win_rates = cumulative_rewards / (np.arange(N) + 1)
        
        plt.figure(figsize=(10, 8))
        plt.plot(win_rates, label="Win Rate")
        plt.plot(np.ones(N)*np.max(bandit_rewards), label='Optimal Win Rate')
        plt.legend()
        plt.title("Win Rate Convergence Thompson Sampling")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.show()

    def plot_bandit_distributions(self, bandits, trial):
        """
        Plots the distribution of each bandit after a given number of trials.
        """
        x = np.linspace(-3, 6, 200)
        for b in bandits:
            y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
            plt.plot(x, y, label = f"real mean: {b.true_mean:.4f}, num plays: {b.N}")
            plt.title(f"Bandit distributions after {trial} trials")
        plt.legend()
        plt.show()
        
    def report(self, bandits, t_rewards, N):

        print(f"Total Reward Earned: {t_rewards.sum()}")
        print(f"Average Reward: {np.mean(t_rewards)}")
        print(f"Overall Win Rate: {t_rewards.sum() / N}")
        print(f"Number of times selected each bandit: {[b.N for b in bandits]}")

# visualisation class is responsible for that
def comparison(epsilon_rewards, thompson_rewards):
    pass
    
if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
