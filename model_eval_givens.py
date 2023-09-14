import math
import fileinput
import numpy as np
from scipy import integrate
from scipy import optimize
import random
import astropy
from astropy import units
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error
#import winsound
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split #splitting data into training and test sets
from sklearn.preprocessing import StandardScaler #feature scaling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from IPython.display import clear_output

from sklearn.metrics import mean_absolute_error, r2_score, f1_score,accuracy_score
from sklearn.linear_model import LinearRegression
#from lightgbm import LGBMRegressor
#from xgboost.sklearn import XGBRegressor
#from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from matplotlib import gridspec
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from numpy import absolute, mean, std
from sklearn.metrics import auc
import sys
import os
from joblib import dump,load
import scipy.stats

model_loc = sys.argv[1]
model = load(model_loc)

version = sys.argv[2]


path = 'npzFiles/'

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
age0 = cosmo.age(0)
tH = age0.to('yr').value

data35 = np.load(path+'filter_35.npz')
data35_filt = data35['filters']

data35_cf = np.load(path+'filter_35_cf.npz')
data35_cf_filt = data35_cf['filters']

data59 = np.load(path+'filter_59.npz')
data59_filt = data59['filters']

data59_cf = np.load(path+'filter_59_cf.npz')
data59_cf_filt = data59_cf['filters']


data35_out = []
data35_cf_out = []
data_59_out = []
data59_cf_out = []

snap35_pred = model.predict(data35_filt)
snap35_cf_pred = model.predict(data35_cf_filt)
snap59_pred = model.predict(data59_filt)
snap59_cf_pred = model.predict(data59_cf_filt)
np.shape(snap35_pred)



t_H = tH*10**-6
binsl = np.arange(0, t_H-1000, 250)
binsl = np.concatenate((binsl,np.arange(t_H-1000, t_H, 100)))
binsl = np.arange(t_H-1000, t_H, 100)
msums, binsedge, binnumber = scipy.stats.binned_statistic(np.linspace(1,100), np.linspace(1,100),
                                                              statistic='sum',bins=binsl)
binsplot = np.repeat(binsedge,2)[1:-1]
#wav = data_filt['wav_filt']


lookback = binsplot-np.max(binsplot)
np.savez(f'snap35_sfh_pred_v{version}.npz',sfh_t = lookback, sfh_val = snap35_pred, gal_label = data35['gal_label'])
np.savez(f'snap35_cf_sfh_pred_v{version}.npz',sfh_t = lookback, sfh_val = snap35_cf_pred, gal_label = data35_cf['gal_label'])
np.savez(f'snap59_sfh_pred_v{version}.npz',sfh_t = lookback, sfh_val = snap59_pred, gal_label = data59['gal_label'])
np.savez(f'snap59_cf_sfh_pred_v{version}.npz',sfh_t = lookback, sfh_val = snap59_cf_pred, gal_label = data59_cf['gal_label'])
