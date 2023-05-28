#!/usr/bin/env python
# coding: utf-8

# # State Action Value Function Example
# 
# In this Jupyter notebook, you can modify the mars rover example to see how the values of Q(s,a) will change depending on the rewards and discount factor changing.

# In[13]:


import numpy as np
from utils import *


# In[14]:


# Do not modify
num_states = 1
num_actions = 3


# In[17]:


terminal_left_reward = 100
terminal_right_reward = 40
each_step_reward = 0

# Discount factor
gamma = .25

# Probability of going in the wrong direction
misstep_prob = 0


# In[18]:


generate_visualization(terminal_left_reward, terminal_right_reward, each_step_reward, gamma, misstep_prob)


# In[ ]:




