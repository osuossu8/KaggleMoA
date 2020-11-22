### General ###
import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Data Wrangling ###
import numpy as np
import pandas as pd
import sys
from scipy import stats

### Machine Learning ###
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA

### Deep Learning ###
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

from pickle import load,dump
from tqdm import tqdm
tqdm.pandas()

sys.path.append("/root/workspace/KaggleMoA")
sys.path.append('iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


train_features = pd.read_csv('inputs/train_features.csv')
train_targets_scored = pd.read_csv('inputs/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('inputs/train_targets_nonscored.csv')

test_features = pd.read_csv('inputs/test_features.csv')
df = pd.read_csv('inputs/sample_submission.csv')

print('done')

