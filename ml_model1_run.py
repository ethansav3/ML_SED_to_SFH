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

all_data = np.load('all_data.npz')
combined_X = np.load('data_x.npy')
combined_y = np.load('data_y.npy')

train_X, test_X, train_y, test_y = train_test_split(combined_X, combined_y, random_state=117, test_size=0.3)
#train_X, test_X, train_y, test_y = train_test_split(combined_X, combined_y, random_state=117)
#-------------------------------------------

#model = LinearRegression()
#model = KNeighborsRegressor()
model = RandomForestRegressor(verbose=0, random_state=1)
#--------------------------------------
print('constructing random forest')
model.fit(train_X, train_y)
print('forest built')
#[mean_squared_error(y_test, y_pred) for y_pred in model.staged_predict(X_test)]
print('evaluating forest')
trainScore = model.score(train_X, train_y)
testScore = model.score(test_X, test_y)
#--------------------------------------
guess_sfh = model.predict(test_X)
#--------------------------------------
mseTest = mean_squared_error(test_y, guess_sfh)
mseTrain = mean_squared_error(train_y, model.predict(train_X))
print(f"Score: {trainScore*100:.1f}% (training), {testScore*100:.1f}% (testing)")
print(f"Mean Squared Error (MSE): {mseTrain:.1e} (training), {mseTest:.1e} (testing)")
#--------------------------------------
# define the evaluation procedure
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
#n_scores = cross_val_score(model, combined_X, combined_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
#n_scores = absolute(n_scores)
# summarize performance
#print('Mean Absolute Error (MAE): %.3f +/- %.3f' % (mean(n_scores), std(n_scores)))
#--------------------------------------





# results plotting

#perm = np.random.permutation(len(gal_nums))
#for idx in range(10):
#for idx in perm[:100]:
#for idx in range(len(gal_nums))[-50:]:
#     plt.figure(figsize=(7,3))
#     gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
#     gnum = gal_nums[idx]
#     zgal = z_gals[idx]
#     fil = filters[idx]
#     wav = wav_filt
#     res = plot_info[f'{zgal}_{gnum}']
     #print(gnum)
     #print(ssfr[idx]*tH)
#     x = np.asarray(res[0])*1e6
#     y = np.asarray(res[1])
#     a1 = auc(x,y) # auc = area under curve (integration)
#     ax = plt.subplot(gs[0])
#     plt.plot(res[0],res[1],label = f'binned true (int: {a1:.1e})')
#     x = np.asarray(res[2])*1e6
#     y = np.asarray(res[3])
#     a2 = auc(x,y) # auc = area under curve (integration)
#     plt.plot(res[2],y,label = f'binned nonpara (int: {a2:.1e})')
     #pred = model.predict(filters[idx])
#     pred = model.predict(filters[idx].reshape(1, -1))
     #pred = model.predict(filters[idx].reshape(-1, 1))
#     y = arr_double(pred)
#     a3 = auc(x,y) # auc = area under curve (integration)
#     plt.plot(res[2],y,label = f'ML predicted (int: {a3:.1e})', ls='--')
#     if(logfit):
#        plt.ylabel('log(SFR+1) [M$_{\odot}/$yr]')
#     else:
#        plt.ylabel('SFR [M$_{\odot}/$yr]')
#     plt.xlabel('t$_H$ [Myr]')
#     prep = None
     #if gnum in trainInds: prep='training'
     #if gnum in testInds: prep='testing'
#     plt.title(f'Galaxy {gnum}, z={zgal}, sSFR*tH: {np.round(ssfr[idx]*tH,5)}, Prep: {prep}')
#     plt.legend(fontsize=9)
#     ax = plt.subplot(gs[1])
#     plt.loglog(wav, fil)
#     plt.xlabel('Wavelength (micron)', size=8)
#     plt.ylabel('Flux')
#     plt.title('SED')
#     plt.tight_layout()
#     plt.show()
#     plt.close()

plt.plot(all_data['wav_filt'],model.feature_importances_)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\lambda$ ($\\mu m$)')
plt.ylabel('Feature Importance')
plt.title('Initial SFH Filter Model Feature Importances')
plt.savefig('m1_feature_importance.png')
plt.close()


from joblib import dump

dump(model,'ml_rf1.joblib')
