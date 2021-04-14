# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:15:01 2021

@author: RK
"""

import numpy as np
import matplotlib.pyplot as plt
def eval_plot(train, valid):
    plt.figure()
    plt.plot(np.arange(len(train)), train, label='Train')
    plt.plot(np.arange(len(valid)), valid, label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")