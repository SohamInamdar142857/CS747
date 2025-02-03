import numpy as np 
from numpy import random
import pandas as pd
import math 
import matplotlib.pyplot as plt 
import pulp as lp
import argparse,time

#Threshold for checking equality 
threshold=1e-8

def file_datatoMDP(file_name):
  file_1=open(file_name,"r+")
  file_1_lines=file_1.readlines()
  count=0
  n_lines=len(file_1_lines)
  lst_filedata=[None]*n_lines
  for count in range(n_lines):
    lst_filedata[count]=file_1_lines[count].split()
  return lst_filedata

def to_policy(file_name):
  if (len(file_name)==0):
    return ''
  file_1=open(file_name,"r+")
  file_1_lines=file_1.readlines()
  count=0
  n_lines=len(file_1_lines)
  policy_arr=[None]*n_lines
  for i in range(n_lines):
    policy_arr[i]=int(file_1_lines[i])
  return np.array(policy_arr,dtype=int)



#Defining a state transition
class transition:
  #This class will store the various parameters for a given transition like initial state, final state, probability and reward.
  def __init__(self,tran_data):
    self.init_state=int(tran_data[1]) 
    self.action=int(tran_data[2]) 
    self.final_state=int(tran_data[3])  
    self.reward=float(tran_data[4]) 
    self.prob=float(tran_data[5])


#Define a finite state MDP
class MDP:
  def __init__(self,data_file):
    #data_file takes string values(file name/file path)
    #Here we store the pre-defined parameters for a given MDP
    self.MDP_data=file_datatoMDP(data_file)
    self.n_states=int(self.MDP_data[0][1]) #number of states
    self.n_actions=int(self.MDP_data[1][1])  #number of actions
    self.end=np.array(self.MDP_data[2][1:],dtype=int)
    self.n_trans=len(self.MDP_data)-5
    self.gamma=float(self.MDP_data[len(self.MDP_data)-1][1]) #gamma/discount factor
    self.typ=self.MDP_data[len(self.MDP_data)-2][1]
    self.all_trans=[self.MDP_data[i] for i in range(3,self.n_trans+3)]  
    self.transitions=np.empty(self.n_trans,dtype=transition)  #data regarding all possible state transitions
    for i in range(self.n_trans):
      self.transitions[i]=transition(self.all_trans[i])

#Initializing the value function V_n(s)
    self.V_curr=np.zeros(self.n_states)
    self.optimal_actions=np.zeros(self.n_states)
    self.optimal_values=np.zeros(self.n_states)
    
  def value_iteration(self):
    for t in range(10000):
      V_next=np.zeros(self.n_states)
      delta=0
      for s in range(self.n_states):
        max_bellman=0
        if s in (self.end):
          self.V_curr[s]=0
          continue
        for a in range(self.n_actions):
          bellman=0
          for trans in range(self.n_trans):
            if ((self.transitions[trans].init_state==s) and
                (self.transitions[trans].action==a)):
              bellman+=self.transitions[trans].prob*(self.transitions[trans].reward+self.gamma*self.V_curr[self.transitions[trans].final_state])
          max_bellman=max(bellman,max_bellman)
          if (self.V_curr[s]<bellman):
            self.optimal_actions[s]=a
        V_next[s]=max_bellman
        delta=max(delta,np.linalg.norm(V_next-self.V_curr))
      self.V_curr=V_next
      if (delta<threshold):
        return self.V_curr,self.optimal_actions
    return self.V_curr,self.optimal_actions
      
  def policy_eval(self,policy):
    while(True):
      V_next=np.zeros(self.n_states)
      delta=0
      for s in range(self.n_states):
        if s in (self.end):
          self.V_curr[s]=0
          continue
        bellman=0
        for trans in range(self.n_trans):
          if ((self.transitions[trans].init_state==s) and
              (self.transitions[trans].action==policy[s])):
            bellman+=self.transitions[trans].prob*(self.transitions[trans].reward+self.gamma*self.V_curr[self.transitions[trans].final_state])
        V_next[s]=bellman
        delta=max(delta,np.linalg.norm(V_next-self.V_curr))
      self.V_curr=V_next
      if (delta<threshold):
        return self.V_curr,policy
      
  def policy_improv(self,policy):
    has_changed=0
    for s in range(self.n_states):
      policy_n=policy[s]
      bellman=np.zeros(self.n_actions)
      for a in range(self.n_actions):
          for trans in range(self.n_trans):
            if ((self.transitions[trans].init_state==s) and
                (self.transitions[trans].action==a)):
              bellman[a]+=self.transitions[trans].prob*(self.transitions[trans].reward+self.gamma*self.V_curr[self.transitions[trans].final_state])
      policy[s]=np.argmax(bellman)
      if (policy_n[s]!=policy[s]):
        has_changed=1
    return policy,has_changed    
      
          

  def howards_pi(self):
    policy=np.random.randint(0,self.n_actions,self.n_states)
    converge=0
    while (converge==0):
      self.V_curr,policy=self.policy_eval(policy)
      policy,has_changed=self.policy_improv(policy)
      if (has_changed==0):
        converge=1

    return self.V_curr,policy
  def lin_prog(self):
    return
  
  def solve(self,policy,algorithm):
    if (len(policy)!=0):
      return self.policy_eval(policy)
    else:
      if (algorithm=="vi"):
        return self.value_iteration()
      elif (algorithm=="hpi"):
        return self.howards_pi()
      elif (algorithm=="lp"):
        return self.lin_prog()
      

    
parser=argparse.ArgumentParser()
parser.add_argument('--mdp', required = True, type = str, help = 'Path to data file')
parser.add_argument('--algorithm', required = False, type = str, help = 'Algorithm', default = 'vi')
parser.add_argument('--policy', required = False, type = str, help = 'Policy file', default = '')

args = parser.parse_args()
mdpfile = args.mdp
algo = args.algorithm
policyfile = args.policy


MDP_fin=MDP(mdpfile)
policy=to_policy(policyfile)
V, policy = MDP_fin.solve(policy,algo)
for i in range(len(policy)):
    print(f'{V[i]:.6f} {policy[i]}')








        
    



                 

            
        

