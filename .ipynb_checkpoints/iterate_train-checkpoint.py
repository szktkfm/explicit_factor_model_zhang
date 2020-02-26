import time

from sklearn.utils import shuffle
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


#class Iterater:
#    def __init__(self, )