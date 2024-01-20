# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# %%
df = pd.read_csv("full_data.csv")
df

TIME = 1077.37

# %% Config Params
# Data Frame
# [0: Full_Data, 1: GreaterThan3, 2: GreaterThan2]
data_source = 2
drop_mds = True
drop_cols = ["Unname    d: 0", "md"]

# Training
cv_size = 0.2

# Data Preprocessing
drop_outliers = True
fix_skewness = True
scale = True
use_pls = False

# Extra configurations
single_run = True
random_state = 42
results_file = "allR2DS.csv"
mse_results_file = "allMSEDS.csv"

# %% Get Data
filename = {0: "full_data.csv", 1: "dataf3.csv", 2: "dataf2.csv"}[data_source]

df = pd.read_csv(filename)
# %%
if drop_mds:
    df = df[df["md"] == 0]

if len(drop_cols) > 0:
    df.drop(drop_cols, axis=1, inplace=True)

df.hist(figsize=(15, 15))

corr = df.corr()

# %% Load Test Data
test_df = pd.read_csv("cross_validation_data.csv")
for col in drop_cols:
    if col in test_df.columns:
        test_df.drop(col, axis=1, inplace=True)
X_test = test_df.drop("time", axis=1)
y_test = test_df["time"]

# %% Data preprocessing
from sklearn import preprocessing
import pickle

if drop_outliers:
    df = df[df["time"] < 500]

predictors = df.drop("time", axis=1)
target = df["time"]

if fix_skewness:
    predictors = predictors ** (1 / 6)
    predictors.hist(figsize=(15, 15))
    print(predictors.skew())
    # Do it for test data
    X_test = X_test ** (1 / 6)

from sklearn.cross_decomposition import PLSRegression

if use_pls:
    pls = PLSRegression(n_components=len(predictors.columns))
    pls.fit(predictors, target)
    predictors = pls.transform(predictors)
    print("Weights are: ", pls.x_weights_)
    # Save transformer
    file = open("PLS_transformer.pickle", "wb")
    # dump information to that file
    pickle.dump(pls, file)
    # close the file
    file.close()

    # Do it for Test Data:
    X_test = pls.transform(X_test)


if scale:
    mean = predictors.mean()
    std = predictors.std()
    predictors = preprocessing.scale(predictors)
    print("Training Mean and STD:", mean, std)

    # Do it for Test Data
    X_test = (X_test - mean) / std

# %% Train-test-cv

print("DataSet size is ", df.shape[0])

from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(
    predictors, target, test_size=cv_size, random_state=random_state
)

targets = {"train": y_train, "test": y_test, "cv": y_cv}

# %% USING DECISION TREES
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    splitter="best",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)

model = model.fit(X_train, y_train)

y_test_predicted = model.predict(X_test)
y_cv_predicted = model.predict(X_cv)
y_train_predicted = model.predict(X_train)


predictions = y_cv_predicted
y_ref = y_cv
ref = np.linspace(min(predictions), max(predictions))
fig = plt.figure(figsize=(10, 8))
plt.scatter(y_ref, predictions)
plt.plot(ref, ref, "k--")
plt.axis("square")
plt.xlabel("y real"), plt.ylabel("y predict")
plt.title(
    "Single Decision Tree RMSE=%0.4f, R^2=%0.4f"%(
        mean_squared_error(y_ref, predictions), r2_score(y_ref, predictions)
    )
)
plt.grid()

print("Train: mse:", mean_squared_error(y_train, y_train_predicted),
      " r2:", r2_score(y_train, y_train_predicted))

print("Test: mse:", mean_squared_error(y_test, y_test_predicted),
      " r2:", r2_score(y_test, y_test_predicted))

print("CV: mse:", mean_squared_error(y_cv, y_cv_predicted),
      " r2:", r2_score(y_cv, y_cv_predicted))

print("Total ERROR:", str(abs(y_test_predicted.sum() - y_test.sum())))

#%%
model = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=None),
                         n_estimators=200,
                         random_state=0,
                         max_samples=0.7,
                         oob_score=False,
                         verbose=1)

model = model.fit(X_train, y_train)

y_test_predicted = model.predict(X_test)
y_cv_predicted = model.predict(X_cv)
y_train_predicted = model.predict(X_train)


predictions = y_cv_predicted
y_ref = y_cv
ref = np.linspace(min(y_ref), max(y_ref))
fig = plt.figure(figsize=(10, 8))
plt.scatter(y_ref, predictions)
plt.plot(ref, ref, "k--")
plt.axis("square")
plt.xlabel("y real"), plt.ylabel("y predict")
plt.title(
    "Bagging with Trees RMSE=%0.4f, R^2=%0.4f"%(
        mean_squared_error(y_ref, predictions), r2_score(y_ref, predictions)
    )
)
plt.grid()

print("Train: mse:", mean_squared_error(y_train, y_train_predicted),
      " r2:", r2_score(y_train, y_train_predicted))

print("Test: mse:", mean_squared_error(y_test, y_test_predicted),
      " r2:", r2_score(y_test, y_test_predicted))

print("CV: mse:", mean_squared_error(y_cv, y_cv_predicted),
      " r2:", r2_score(y_cv, y_cv_predicted))

print("Total ERROR:", str(abs(y_test_predicted.sum() - y_test.sum())))

#%%
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
                         n_estimators=2000,
                         random_state=0,
                         max_samples=0.7,
                         oob_score=False,
                         verbose=1)

model = model.fit(X_train, y_train)

y_test_predicted = model.predict(X_test)
y_cv_predicted = model.predict(X_cv)
y_train_predicted = model.predict(X_train)


predictions = y_cv_predicted
y_ref = y_cv
ref = np.linspace(min(y_ref), max(y_ref))
fig = plt.figure(figsize=(10, 8))
plt.scatter(y_ref, predictions)
plt.plot(ref, ref, "k--")
plt.axis("square")
plt.xlabel("y real"), plt.ylabel("y predict")
plt.title(
    "Random Forest RMSE=%0.4f, R^2=%0.4f"%(
        mean_squared_error(y_ref, predictions), r2_score(y_ref, predictions)
    )
)
plt.grid()

print("Train: mse:", mean_squared_error(y_train, y_train_predicted),
      " r2:", r2_score(y_train, y_train_predicted))

print("Test: mse:", mean_squared_error(y_test, y_test_predicted),
      " r2:", r2_score(y_test, y_test_predicted))

print("CV: mse:", mean_squared_error(y_cv, y_cv_predicted),
      " r2:", r2_score(y_cv, y_cv_predicted))

print("Total ERROR:", str(abs(y_test_predicted.sum() - y_test.sum())))

# %%
x

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Train: mse:", mean_squared_error(y_train, y_train_predicted),
      " r2:", r2_score(y_train, y_train_predicted))

print("Test: mse:", mean_squared_error(y_test, y_test_predicted),
      " r2:", r2_score(y_test, y_test_predicted))

print("CV: mse:", mean_squared_error(y_cv, y_cv_predicted),
      " r2:", r2_score(y_cv, y_cv_predicted))

#%%
model = BaggingRegressor(SVR(kernel="poly", epsilon=10, degree=5, coef0=1),
                              n_estimators=100,
                              random_state=0,
                              max_samples=0.8,
                              oob_score=False,
                              verbose=1
                              )
model.fit(X_train, y_train)

y_test_predicted = model.predict(X_test)
y_cv_predicted = model.predict(X_cv)
y_train_predicted = model.predict(X_train)


predictions = y_cv_predicted
y_ref = y_cv
ref = np.linspace(min(y_ref), max(y_ref))
fig = plt.figure(figsize=(10, 8))
plt.scatter(y_ref, predictions)
plt.plot(ref, ref, "k--")
plt.axis("square")
plt.xlabel("y real"), plt.ylabel("y predict")
plt.title(
    "Random Forest RMSE=%0.4f, R^2=%0.4f"%(
        mean_squared_error(y_ref, predictions), r2_score(y_ref, predictions)
    )
)
plt.grid()

print("Train: mse:", mean_squared_error(y_train, y_train_predicted),
      " r2:", r2_score(y_train, y_train_predicted))

print("Test: mse:", mean_squared_error(y_test, y_test_predicted),
      " r2:", r2_score(y_test, y_test_predicted))

print("CV: mse:", mean_squared_error(y_cv, y_cv_predicted),
      " r2:", r2_score(y_cv, y_cv_predicted))

print("Total ERROR:", str(abs(y_test_predicted.sum() - y_test.sum())))


#%%
