#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:47:58 2019

@author: lhjin

Modified on Fri Oct 12:19:00 2021

@author: Bereket Kebede

Modified on Fri Nov 10:23:00 2021
"""


import os
import math
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt

path = 'C:/Users/bkebede/OneDrive - The University of Memphis/Documents/GitHub/DeepLearning/Testing_codes/UNet'
sys.path.append(path)


print(torch.cuda.is_available())
from unet_model import UNet
import warnings
warnings.filterwarnings('ignore')


cuda = torch.device('cuda:0')
model = UNet(n_channels=3, n_classes=1)
model.cuda(cuda)
#model.load_state_dict(torch.load("D:/Bereket/DeepLearning/Training_codes/UNet/UNet_SIM3_microtubule.pkl"))
net = torch.load("D:/Bereket/DeepLearning/Training_codes/UNet/Luhong_Model/UNet_SIM3_microtubule.pkl")

plt.plot(net)
print("plot finished")
    
