"""
You need to write code to plot the graphs as required in task2 of the problem statement:
    - You can edit any code in this file but be careful when modifying the simulation specific code. 
    - The simulation framework as well as the BernoulliBandit implementation for this task have been separated from the rest of the assignment code and is contained solely in this file. This will be useful in case you would like to collect more information from runs rather than just regret.
"""

import numpy as np
from multiprocessing import Pool
from task1 import Eps_Greedy, UCB, KL_UCB
import matplotlib.pyplot as plt
# START EDITING HERE
# You can use this space to define any helper functions that you need.
# END EDITING HERE

class BernoulliArmTask2:
  def __init__(self, p):
    self.p = p

  def pull(self, num_pulls=None):
    return np.random.binomial(1, self.p, num_pulls)

class BernoulliBanditTask2:
  def __init__(self, probs=[0.3, 0.5, 0.7],):
    self.__arms = [BernoulliArmTask2(p) for p in probs]
    self.__max_p = max(probs)
    self.__regret = 0

  def pull(self, index):
    reward = self.__arms[index].pull()
    self.__regret += self.__max_p - reward
    return reward

  def regret(self):
    return self.__regret
  
  def num_arms(self):
    return len(self.__arms)


def single_sim_task2(seed=0, ALGO=Eps_Greedy, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
  np.random.seed(seed)
  np.random.shuffle(PROBS)
  bandit = BernoulliBanditTask2(probs=PROBS)
  algo_inst = ALGO(num_arms=len(PROBS), horizon=HORIZON)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.regret()

def simulate_task2(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim_task2,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  regrets = np.mean(sim_out)

  return regrets

def task2(algorithm, horizon, p1s, p2s, num_sims=50):
    """generates the data for task2
    """
    probs = [[p1s[i], p2s[i]] for i in range(len(p1s))]

    regrets = []
    for prob in probs:
        regrets.append(simulate_task2(algorithm, prob, horizon, num_sims))

    return regrets

if __name__ == '__main__':
  # EXAMPLE CODE
  '''task2p1s = [0.2, 0.3]
  task2p2s = [0.1, 0.2]
  regrets = task2(Eps_Greedy, 1000, task2p1s, task2p2s, 1)
  print(regrets)'''
  #Part A-UCB Algorithm
  p1_const=0.9
  step_val=0.05
  n_steps=int((p1_const/step_val)+1)
  p1arr=np.zeros(n_steps)
  p2arr=np.zeros(n_steps)
  for i in range(n_steps):
    p1arr[i]=0.9
    p2arr[i]=0.05*i
  regrets_taskA=task2(UCB,30000,p1arr,p2arr,1)
  print(regrets_taskA)
  plt.plot(p2arr,regrets_taskA,color="orange")
  plt.title("Average Regret vs p2")
  plt.xlabel("p2(Variable probability arm)")
  plt.ylabel("Average regret")
  plt.grid(True)
  plt.show()
  #Part B-UCB vs KL-UCB
  p_max=0.9
  step_val=0.05
  n_steps=int((p_max/step_val)+1)
  p1arr=np.zeros(n_steps)
  p2arr=np.zeros(n_steps)
  for i in range(n_steps):
    p1arr[i]=0.1+(0.05*i)
    p2arr[i]=0.05*i
  regrets_taskB_UCB=task2(UCB,30000,p1arr,p2arr,1)
  regrets_taskB_KLUCB=task2(KL_UCB,30000,p1arr,p2arr)
  #Plot for UCB
  plt.plot(p2arr,regrets_taskB_UCB,color="blue", label="UCB")
  plt.legend()
  plt.title("Average Regret vs p2")
  plt.xlabel("p2")
  plt.ylabel("Average regret")
  plt.grid(True)
  plt.show()

  #Plot for KL-UCB
  plt.plot(p2arr,regrets_taskB_KLUCB,color="red", label="KL-UCB")
  plt.legend()
  plt.title("Average Regret vs p2")
  plt.xlabel("p2")
  plt.ylabel("Average regret")
  plt.grid(True)
  plt.show()
  # INSERT YOUR CODE FOR PLOTTING HERE
  pass