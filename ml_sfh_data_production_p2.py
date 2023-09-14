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
import scipy.stats
import pandas as pd


if(int(sys.argv[1])==1):
    logfit=True
    print('fitting log(sfr+1)')
else:
    logfit=False
    print('fitting sfr')

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
age0 = cosmo.age(0)

print('loading complete dataset')

full_list = np.load('all_data.npz')

stellar_masses = full_list['stellar_mass'] # solar masses
dust_masses = full_list['dust_mass'] # solar masses
sfrs = full_list['sfr'] # msun/yr
metal = full_list['metallicity'] # individual, not scaled bto sun
gal_nums = full_list['gal_num'] # number assigned by group code
z_gals = full_list['z']
filters = full_list['filters']*units.erg/units.s/units.cm**2/units.micron
wav_filt = full_list['wav_filt'] # micron

print('dataset loaded')

ssfr = sfrs/stellar_masses
tH = age0.to('yr').value

age_arr = cosmo.age(z_gals).to('yr').value
cut1 = ssfr > 1/age_arr
cut2 = ssfr < 0.2/age_arr

binwidth = 100 #Myr
print(f'main binwidth = {binwidth} Myr')
def get_galaxy_SFH_fit_style(massform,tform,nbins = 10):
    t_H = tH*10**-6
    tbinmax = (t_H * 0.8) #* 1e9 #earliest time bin goes from age = 0 to age = 2.8 Gyr
    lim1, lim2 = 8.0-6,8.477-6#7.47, 8.0 #most recent time bins at 100 Myr and 300 Myr ago
    agelims = [0,lim1]
    agelims += np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist()
    agelims += [np.log10(t_H)]#*1e9)]
      #print(agelims)
    agebins = np.array([agelims[:-1], agelims[1:]])
    #print(agebins)
    #print(10**agebins)
    #binsl =(t_H-10**np.array(agelims))[::-1]
    #binsl = np.arange(0, t_H-1000, 250)
    #binsl = np.concatenate((binsl,np.arange(t_H-1000, t_H, 100)))
    #binsl = np.arange(t_H-1000, t_H, 100)
    binsl = np.arange(0,t_H,500)
    #print(tH, agelims, binsl, '\n')
    #print(binsl)
    msums, binsedge, binnumber = scipy.stats.binned_statistic(tform, massform,
                                                              statistic='sum',bins=binsl)
    msums[np.isnan(msums)] = 0
    timediff = np.diff(binsedge)
    sfr_fit = msums/timediff/10**6
    sfr_plot = []
    time_plot = []
    for i in range(len(sfr_fit)):
        time_plot.append(binsedge[i])
        time_plot.append(binsedge[i+1])
        sfr_plot.append(sfr_fit[i])
        sfr_plot.append(sfr_fit[i])
    return time_plot,sfr_plot,sfr_fit


#this function is given to scipy.stats.binned_statistics
#and acts on the particle masses per bin to give total(Msun) / timebin
def get_massform(massform):
        return np.sum(massform) / (binwidth * 1e6)
def get_galaxy_SFH(file_,galaxy_id,galz):
    dat = pd.read_pickle(file_)
    massform = np.array(dat['massform'][np.where(np.asarray(dat['id'])==galaxy_id)[0][0]])
    tform =  np.array(dat['tform'][np.where(np.asarray(dat['id'])==galaxy_id)[0][0]])*1000
    age_univ = cosmo.age(galz).to('Myr').value
    t0 = cosmo.age(0).to('Myr').value

    tform = tform+(t0-age_univ)


    #convert from Gyr to Myr
    t_H = np.max(tform)
    #bins = np.arange(np.min(tform), np.max(tform), binwidth) #can use whatever bin size/start/end that fit your problem
    bins = np.arange(0, t_H, binwidth) #can use whatever bin size/start/end that fit your problem
    #print(bins)
    sfrs, bins, binnumber = scipy.stats.binned_statistic(tform, massform,
                                                         statistic=get_massform, bins=bins)
    sfrs[np.isnan(sfrs)] = 0
    bincenters = 0.5*(bins[:-1]+bins[1:])
    sfh = sfrs
   # get_galaxy_SFH_fit_style(massform,tform)
    # x,y,z = get_galaxy_SFH_fit_style(massform,tform,nbins = 10)
    x,y,z = get_galaxy_SFH_fit_style(massform,tform)
    return bincenters, sfh, x, y, z

def arr_double(arr):
    #print(np.shape(arr))
    new_arr = []
    try:
        for val in arr[0]:
          new_arr.append(val)
          new_arr.append(val)
    except:
        for val in arr:
          new_arr.append(val)
          new_arr.append(val)
    return np.array(new_arr)



#path = paths[user]+'/snap305_sfhs.pickle'
z_dict = {0:305,1:212,2:160,3:127,4:104,5:87,6:74,3.5:22}
plot_info = {}
sfh_to_fit = []
paths = {0:'/orange/narayanan/d.zimmerman/simba/m25n512/snap305/snap305_sfhs.pickle',
        1:'/orange/narayanan/d.zimmerman/simba/m25n512/snap212/snap212_sfhs.pickle',
        2:'/orange/narayanan/d.zimmerman/simba/m25n512/snap160/snap160_sfhs.pickle',
        3:'/orange/narayanan/d.zimmerman/simba/m25n512/snap127/snap127_sfhs.pickle',
        4:'/orange/narayanan/d.zimmerman/simba/m25n512/snap104/snap104_sfhs.pickle',
        5:'/orange/narayanan/d.zimmerman/simba/m25n512/snap87/snap87_sfhs.pickle',
        6:'/orange/narayanan/d.zimmerman/simba/m25n512/snap74/snap74_sfhs.pickle',
        3.5:'/orange/narayanan/d.zimmerman/smuggle/ml_m25n512b/snap22_sfhs.pickle'}
print('constructing sfh dataset')

for i in range(len(gal_nums)):
    galaxy_id = gal_nums[i]
    gal_z = z_gals[i]
    print(f'z={gal_z},galaxy {galaxy_id}')
    time, sfh, time_prosp,sfh_prosp, sfr_fit = get_galaxy_SFH(paths[gal_z],galaxy_id,gal_z)

    if(logfit):
        plot_info[f'{gal_z}_{galaxy_id}'] = [time, np.log10(np.array(sfh)+1), time_prosp,np.log10(np.array(sfh_prosp)+1)]
        sfh_to_fit.append(np.log10(np.array(sfr_fit)+1))
    else:
        plot_info[f'{gal_z}_{galaxy_id}'] = [time, np.array(sfh), time_prosp,np.array(sfh_prosp)]
        sfh_to_fit.append(np.array(sfr_fit))

print('sfh complete dataset built')
combined_y = np.array(sfh_to_fit)
combined_X = np.array(filters)
print(np.shape(combined_y))
print(np.shape(combined_X))

print('saving relevant data')
np.save('data_x.npy',combined_X)
np.save('data_y.npy',combined_y)

np.save('sfh_plot_info.npy',plot_info)
