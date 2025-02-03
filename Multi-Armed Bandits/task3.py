"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault # probability that the bandit returns a faulty pull
        # START EDITING HERE
        self.success=np.zeros(num_arms)
        self.failures=np.zeros(num_arms)
        self.beta_samples=np.zeros(num_arms)
        
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        #Same as Thompson Sampling
        for i in range(self.num_arms):
            self.beta_samples[i]=np.random.beta(self.success[i]+1,self.failures[i]+1)
        return np.argmax(self.beta_samples)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.success[arm_index]+=reward
        self.failures[arm_index]+=(1-reward)
        #END EDITING HERE

