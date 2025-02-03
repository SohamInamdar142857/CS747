"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need

def kl(p,q):
    if(q==1):
        return 10e7
    elif (p==0):
        return (math.log(1/(1-q)))
    elif (p==1):
        return math.log(1/q)
    else:
        return (p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q)))
    
def bin_search(time,c,p,n_pulls):
    ub=(math.log(time)+c*math.log(math.log(time)))/n_pulls
    l=p #lower limit of search space
    r=1 #upper limit of search space
    while(r-l>0.001):
        mid=(l+r)/2
        if (kl(p,mid)<ub):
            l=mid
        else:
            r=mid
        
    return (l+r)/2;   

        
    
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucb=np.zeros(num_arms)
        self.time=0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if(self.time<self.num_arms):
            return self.time

        else:
            for i in range(self.num_arms):
                
                self.ucb[i]=self.values[i]+(math.sqrt(2*math.log(self.time)/self.counts[i]))

        
            return np.argmax(self.ucb)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        self.values[arm_index]+=(reward-self.values[arm_index])/(self.counts[arm_index])
        
        self.time+=1
       
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        #empirical reward
        self.values = np.zeros(num_arms)
        self.c=0
        self.time=0
        self.kl = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        #t=0 to num_arms
        if(self.time<self.num_arms):
            return self.time

        #after t=num_arms+1
        else:
            #Calculate kl values for all arms
            for i in range(self.num_arms):
                self.kl[i]=bin_search(self.time,self.c,self.values[i],self.counts[i])
            
            return np.argmax(self.kl)

            

        
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        self.values[arm_index]+=(reward-self.values[arm_index])/(self.counts[arm_index])
        
        
        self.time+=1
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.success=np.zeros(num_arms)
        self.failures=np.zeros(num_arms)
        self.beta_samples=np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for i in range(self.num_arms):
            self.beta_samples[i]=np.random.beta(self.success[i]+1,self.failures[i]+1)
        return np.argmax(self.beta_samples)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.success[arm_index]+=reward
        self.failures[arm_index]+=(1-reward)
        # END EDITING HERE
