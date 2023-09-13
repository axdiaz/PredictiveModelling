# -*- coding: utf-8 -*-
''' 
Example of decomposition by the PLS method (Partial Least Square) 
'''

#%% Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

#%%  Create or upload data for analysis
data = pd.read_csv('Data/datos_PLS.txt')
# data = pd.read_csv('Data/datos_PLS2.txt')

#%% Data visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.x1, data.x2, data.x3)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.show()

#%% Using Seaborn library to compare outputs against inputs
import seaborn as sns
fig = sns.pairplot(data,x_vars=['x1','x2','x3'],
             y_vars=['y'])
# fig.savefig('../figures/P2_fig/F2.png')

#%% Principal component analysis application
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(data[['x1','x2','x3']])
data_pca = pca.transform(data[['x1','x2','x3']])
data_pca = pd.DataFrame(data_pca,columns=['x1*','x2*','x3*'])
data_pca['y'] = data['y']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca['x1*'], data_pca['x2*'], data_pca['x3*'])
ax.set_xlabel('X1*')
ax.set_ylabel('X2*')
ax.set_zlabel('X3*')
plt.show()
# fig.savefig('../figures/P2_fig/F3.png')

fig = sns.pairplot(data_pca,x_vars=['x1*','x2*','x3*'],
             y_vars=['y'])
# fig.savefig('../figures/P2_fig/F4.png')

#%% Application of linear regression with original data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,r2_score)

X_train, X_test, y_train, y_test = train_test_split(data[['x1','x2','x3']], data.y,
                                                    test_size=0.2, random_state=42)
linreg = LinearRegression()
linreg.fit(X_train, y_train)

ref = np.linspace(min(y_test),max(y_test))

y_predict = linreg.predict(X_test)
fig = plt.figure(figsize=(10,8))
plt.scatter(y_test,y_predict)
plt.plot(ref,ref,'k--')
plt.axis('square')
plt.xlabel('y real'),plt.ylabel('y predict')
plt.title('Linear regression (original), RMSE=%0.4f, R^2=%0.4f'%(mean_squared_error(y_test,y_predict),r2_score(y_test,y_predict)))
plt.grid()
# fig.savefig('../figures/P2_fig/F5.png')

#%% Application of linear regression with PCA data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(data_pca[['x1*','x2*','x3*']], data_pca.y,
                                                    test_size=0.2, random_state=42)
linreg = LinearRegression()
# linreg.fit(np.array(X_train['x1*']).reshape(80,1), y_train) # case one-dimensional 
# linreg.fit(np.array(X_train[['x1*','x2*']]), y_train) # case two-dimensional 
linreg.fit(np.array(X_train[['x1*','x2*','x3*']]), y_train) # case three-dimensional

ref = np.linspace(min(y_test),max(y_test))

# y_predict = linreg.predict(np.array(X_test['x1*']).reshape(20,1)) # case one-dimensional 
# y_predict = linreg.predict(np.array(X_test[['x1*','x2*']])) # case two-dimensional
y_predict = linreg.predict(np.array(X_test[['x1*','x2*','x3*']])) # case three-dimensional
fig = plt.figure(figsize=(10,8))
plt.scatter(y_test,y_predict)
plt.plot(ref,ref,'k--')
plt.axis('square')
plt.xlabel('y real'),plt.ylabel('y predict')
plt.title('PCA regression, RMSE=%0.4f, R^2=%0.4f'%(mean_squared_error(y_test,y_predict),r2_score(y_test,y_predict)))
plt.grid()
# fig.savefig('../figures/P2_fig/F6.png')

#%% Apply PLS decomposition and visualize obtained components
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[['x1','x2','x3']], data.y,
                                                    test_size=0.2, random_state=42)
pls = PLSRegression(n_components=3)
pls.fit(X_train, y_train)
data_pls = pls.transform(data[['x1','x2','x3']])
data_pls = pd.DataFrame(data_pls,columns=['x1*','x2*','x3*'])
data_pls['y'] = data['y']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pls['x1*'], data_pls['x2*'], data_pls['x3*'])
ax.set_xlabel('X1*')
ax.set_ylabel('X2*')
ax.set_zlabel('X3*')
plt.show()
# fig.savefig('../figures/P2_fig/F7.png')

fig = sns.pairplot(data_pls,x_vars=['x1*','x2*','x3*'],
             y_vars=['y'])
# fig.savefig('../figures/P2_fig/F8.png')


#%% Apply linear regression for PLS data

pls = PLSRegression(n_components=2)
pls.fit(X_train, y_train)
y_predict = pls.predict(X_test)

'''
T: x_scores_
U: y_scores_
W: x_weights_
C: y_weights_
P: x_loadings_
Q: y_loadings_
'''

fig = plt.figure(figsize=(10,8))
plt.scatter(y_test,y_predict)
plt.plot(ref,ref,'k--')
plt.axis('square')
plt.xlabel('y real'),plt.ylabel('y predict')
plt.title('PLS regression, RMSE=%0.4f, R^2=%0.4f'%(mean_squared_error(y_test,y_predict),r2_score(y_test,y_predict)))
plt.grid()
# fig.savefig('../figures/P2_fig/F9.png')


#%% Calculation of coefficients of linear regression using PLS
P = pls.x_loadings_
Q = pls.y_loadings_
W = pls.x_weights_
# Betas = W*(P^T*W)^-1*Q^T
Betas = np.dot(np.dot(P,np.linalg.inv(np.dot(P.T,P))),Q.T)
# Beta0 =  Q_0-P_0.T*Betas
Beta0 = Q[:,0]-np.dot(P[:,0].T,Betas)

#%% Evaluation of transformations
corr_origin = np.corrcoef(data,rowvar=False)
corr_pca = np.corrcoef(data_pca,rowvar=False)
corr_pls = np.corrcoef(data_pls,rowvar=False)
#%%
