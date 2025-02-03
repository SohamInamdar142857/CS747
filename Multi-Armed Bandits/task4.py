"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np
import math
# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE


class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.b1_suc=np.zeros(num_arms)
        self.b2_suc=np.zeros(num_arms)
        self.b1_fail=np.zeros(num_arms)
        self.b2_fail=np.zeros(num_arms)
        #Compute a parameter based on which the decision will be taken
        self.dec_param=np.zeros(num_arms)
        self.horizon = horizon
        self.time=1
        # START EDITING HERE
        
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for i in range(self.num_arms):
            self.dec_param[i]=np.random.beta(self.b1_suc[i]+self.b2_suc[i]+1,self.b1_fail[i]+self.b2_fail[i]+1)       
        return np.argmax(self.dec_param)
        # END EDITING HERE
    
    def get_reward(self, arm_index, set_pulled, reward):
        # START EDITING HERE
        if (set_pulled==0):
            self.b1_suc[arm_index]+=reward
            self.b1_fail[arm_index]+=(1-reward)
        else:
            self.b2_suc[arm_index]+=reward
            self.b2_fail[arm_index]+=(1-reward)
        #self.time+=1
        # END EDITING HERE

