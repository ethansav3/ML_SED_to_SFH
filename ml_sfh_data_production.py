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


path = 'npzFiles/'


file_list = [305,212,160,127,104,87,74,22]
z_list = [0,1,2,3,4,5,6,3.5]
data_list = []
print('loading npzs')
for f in file_list:
  data_filt = np.load(path+f'all_filter_snap{f}.npz')
  data_list.append(data_filt)
for key in data_filt.keys():
    print(key)
print('npzs loaded')
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
age0 = cosmo.age(0)

full_list = {}
full_list['filters'] = []
full_list['gal_num'] = []
full_list['stellar_mass'] = []
full_list['dust_mass'] = []
full_list['metallicity'] = []
full_list['sfr'] = []
full_list['z'] = []

print('constructing complete dictionary')
for i in range(len(data_list)):
  dic = data_list[i]
  z = z_list[i]
  for j in range(len(dic['filters'])):
    full_list['filters'].append(dic['filters'][j])
    full_list['gal_num'].append(dic['gal_num'][j])
    full_list['stellar_mass'].append(dic['stellar_mass'][j])
    full_list['dust_mass'].append(dic['dust_mass'][j])
    full_list['metallicity'].append(dic['metallicity'][j])
    full_list['sfr'].append(dic['sfr'][j])
    full_list['z'].append(z)

full_list['filters'] = np.array(full_list['filters'])
full_list['stellar_mass'] = np.array(full_list['stellar_mass'])
full_list['dust_mass'] = np.array(full_list['dust_mass'])
full_list['metallicity'] = np.array(full_list['metallicity'])
full_list['sfr'] = np.array(full_list['sfr'])
full_list['z'] = np.array(full_list['z'])
full_list['gal_num'] = np.array(full_list['gal_num'])

print('saving completed dictionary')
np.savez('all_data.npz',filters = full_list['filters'],stellar_mass = full_list['stellar_mass'],dust_mass = full_list['dust_mass'],metallicity = full_list['metallicity'],sfr = full_list['sfr'],z = full_list['z'],gal_num = full_list['gal_num'],wav_filt=data_list[0]['wav_filt'])
