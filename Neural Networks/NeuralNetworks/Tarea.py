#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv("energydata_complete.csv")
df.describe()

inputs = df.drop(["Appliances", "lights", "date", "rv1", "rv2"], axis=1)
target = df["Appliances"] + df["lights"]
target.name = "total_energy"

#%%
inputs.hist(figsize=(20,20))
inputs.skew()[np.abs(inputs.skew()) > 1.5]


# In[8]:


# RH5 seems like the only one with relevant skewness, so we are transforming it with a power
inputs["RH_5"] = inputs["RH_5"] ** (1/3)
inputs["RH_5"].skew()


# ## Split Train Test

# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs, target,
                                                    test_size=0.2, random_state=42)


# ### Transform using normalization

# In[10]:


original_mean = X_train.mean()
original_std = X_train.std()


# In[11]:


from sklearn import preprocessing
X_train = preprocessing.scale(X_train)


# In[12]:


X_test = (X_test - original_mean) / original_std


# # Neural Network Exploration

# In[13]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import SGD, Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate dummy data
import numpy as np
from sklearn.metrics import r2_score


# In[79]:

model = Sequential(
    [
        tf.keras.Input(shape=(24,), name="input_layer"),
        Dense(72, activation="tanh"),
        Dense(72, activation="tanh"),
        Dense(1, activation="linear", name="output_layer")
    ]
)


# In[80]:


# optimizer = SGD(learning_rate=0.0001, momentum=0.05)
optimizer = Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-7)


# In[81]:


model.compile(optimizer=optimizer, loss="mean_squared_error")


# In[82]:


r2_train = []
r2_test = []


# In[83]:


from datetime import datetime
for i in range(1):
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    history = model.fit(X_train, y_train, epochs=1000, batch_size=100)
    score_train = model.evaluate(X_train, y_train)
    Yhat_train = model.predict(X_train)
    
    r2_train.append(r2_score(y_train, Yhat_train))
    
    score_test = model.evaluate(X_test, y_test)
    Yhat_test = model.predict(X_test)
    r2_test.append(r2_score(y_test, Yhat_test))
    
    dt = str(datetime.now())
    base_name = "models/model_" + str(i) + "_score_test_" + str(np.round(r2_test[i], 4)) + "__" + dt
    with open(base_name + ".json", "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(base_name + ".h5")
    print(r2_train[i], ":::", r2_test[i])


# %% Visualize the evolution of learning
J = history.history["loss"]
fig = plt.figure(figsize=(15, 8))
plt.plot(np.arange(len(J)), J, linewidth=3)
plt.title("J(W) vs epoch: J(W)=%0.4f" % J[-1] + " R2: %0.4f" % r2_score(y_train, Yhat_train))
plt.xlabel("epoch")
plt.ylabel("J(W)")
plt.grid()
plt.show()


# In[86]:


fig = plt.figure(figsize=(15, 8))
plt.scatter(y_train, Yhat_train)
plt.plot(y_train, y_train, "k--", linewidth=3)
plt.axis("square")
plt.title("Train Prediction: R^2=%0.4f" %  r2_score(y_train, Yhat_train))
plt.xlabel("Y")
plt.ylabel("Yhat")
plt.grid()
plt.show()


# In[87]:


fig = plt.figure(figsize=(15, 8))
plt.scatter(y_test, Yhat_test)
plt.plot(y_test, y_test, "k--", linewidth=3)
plt.axis("square")
plt.title("Prediction: R^2=%0.4f" % r2_score(y_test, Yhat_test))
plt.xlabel("Y")
plt.ylabel("Yhat")
plt.grid()
plt.show()


# In[149]:


from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers  import KerasRegressor

def create_model(lr=0.1, beta=0.9 ,activation="sigmoid"):
    # Neural network architecture
    model = Sequential()
    model.add(tf.keras.Input(shape=(24,), name="input_layer"))
    
    model.add(Dense(72,activation=activation))
    model.add(Dense(72,activation=activation))
        
    model.add(Dense(1,activation='linear'))
    # Optimizer configuration
    opt = Adam(learning_rate=lr, beta_1=beta)
    model.compile(loss = 'mean_squared_error',
                  optimizer=opt,
                  metrics=['mse'])
    return model




# In[54]:


from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

# define the grid search parameters
learning_rates = [0.0001, 0.00005, 0.001]
betas = [0.9, 0.99, 0.8]
activation = ["sigmoid", "tanh", "relu"]
param_grid = dict(lr=learning_rates, beta=betas,activation=activation)

selection_score = make_scorer(r2_score)


# 

# In[55]:


model_search = KerasRegressor(build_fn=create_model,epochs=500, activation="sigmoid", beta=0.9 ,lr=0.1, batch_size=100)


# In[56]:


grid = GridSearchCV(estimator=model_search, param_grid=param_grid,
                    cv=3,return_train_score=True,
                    scoring=selection_score)


# 

# In[ ]:


grid_result = grid.fit(X_train, y_train, )


# In[58]:


print("Best parameters found:", grid_result.best_params_)
print("Best estimator found:", grid_result.best_estimator_)

# Extract mean test scores and standard deviations
mean_scores = grid_result.cv_results_['mean_test_score']
std_scores = grid_result.cv_results_['std_test_score']

for mean_score, std_score, params in zip(mean_scores, std_scores, grid_result.cv_results_['params']):
    print(f"Mean: {mean_score:.4f}, Std: {std_score:.4f} with: {params}")


# In[68]:


grid_result.best_estimator_


# In[66]:


grid_result.best_score_



# In[ ]:




