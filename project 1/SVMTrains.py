from typing import Dict, Any

import pandas as pd
import numpy as np

# %% Config Params
# Data Frame
# [0: Full_Data, 1: GreaterThan3, 2: GreaterThan2]
data_source = 2
drop_mds = True
drop_cols = ["Unnamed: 0", "md"]

# SVM Paramaters
# Kernels: poly, rbf, sigmoid
kernel = "poly"
epsilons = [5]
degrees = [1,2,3,5,6,7,8]  # For Poly
gammas = [2,3,5,10,15]  # For rbf (radial)


# Training
test_size = 0.2
use_cv = True

# Data Preprocessing
drop_outliers = True
fix_skewness = True
scale = True
use_pls = True

# Extra configurations
single_run = True
random_state = 42

# %% Get Data
filename = {0: "full_data.csv", 1: "dataf3.csv", 2: "dataf2.csv"}[data_source]

df = pd.read_csv(filename)
# %%
if drop_mds:
    df = df[df["md"] == 0]

if len(drop_cols) > 0:
    df.drop(drop_cols, axis=1, inplace=True)

df.hist(figsize=(15,15))

corr = df.corr()


#%% Data preprocessing
from sklearn import preprocessing

if drop_outliers:
    df = df[df["time"] < 500]

predictors = df.drop("time", axis=1)
target = df["time"]

if fix_skewness:
    predictors = predictors ** (1/6)
    predictors.hist(figsize=(15,15))
    print(predictors.skew())

from sklearn.cross_decomposition import PLSRegression

if use_pls:
    pls = PLSRegression(n_components=len(predictors.columns))
    pls.fit(predictors, target)
    predictors = pls.transform(predictors)

if scale:
    predictors = preprocessing.scale(predictors)

# %% Train-test-cv



print("DataSet size is ", df.shape[0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    predictors, target, test_size=test_size, random_state=random_state
)

if use_cv:
    cv_df = pd.read_csv("cross_validation_data.csv")
    for col in drop_cols:
        if col in cv_df.columns:
            cv_df.drop(col, axis=1, inplace=True)
    X_cv = cv_df.drop("time", axis=1)
    if fix_skewness:
        X_cv = X_cv ** (1/6)
    if use_pls:
        X_cv = pls.transform(X_cv)
    if scale:
        X_cv = preprocessing.scale(X_cv)

    y_cv = cv_df["time"]

targets = {
    "train": y_train,
    "test": y_test,
    "cv": y_cv
}

#%%


def plot_predictions(preds, description, sources=["train", "test", "cv"]):
    fig, ax = plt.subplots()
    for source in sources:
        y = targets[source]
        pred = preds[source]
        ref = np.linspace(min(y), max(y))
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(y, pred)
        plt.plot(ref, ref, "k--")
        plt.axis("square")
        plt.xlabel("y real"), plt.ylabel("y predict")
        plt.title(
            "{}:{} , RMSE=%0.4f, R^2=%0.4f".format(description, source, mean_squared_error(y, pred), r2_score(y, pred))
        )
        plt.grid()


# %%
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


all_mse = pd.DataFrame()
all_r2 = pd.DataFrame()
all_preds = {}

def get_metrics(description, preds):
    mse_row = {
        "model": description,
        "train": mean_squared_error(y_train, preds["train"]),
        "test": mean_squared_error(y_test, preds["test"]),
    }
    if use_cv:
        mse_row["cv"] = mean_squared_error(y_cv, preds["cv"])
    r2_row = {
        "model": description,
        "train": r2_score(y_train, preds["train"]),
        "test": r2_score(y_test, preds["test"]),
    }
    if use_cv:
        r2_row["cv"] = r2_score(y_cv, preds["cv"])
    return mse_row, r2_row


def get_predictions(model):
    preds = {
        "train": model.predict(X_train),
        "test": model.predict(X_test),
    }
    if use_cv:
        preds["cv"] = model.predict(X_cv)
    return preds


# Train Base Linear Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
preds = get_predictions(linear_reg)
mse, r2 = get_metrics("linear regression", preds)
all_mse = all_mse.append(mse, ignore_index=True)
all_r2 = all_r2.append(r2, ignore_index=True)
all_preds["linear"] = preds

# Train SVMs
if kernel == "poly":
    for degree in degrees:
        for epsilon in epsilons:
            model_svr = SVR(kernel=kernel, epsilon=epsilon, degree=degree, coef0=1)
            model_svr.fit(X_train, y_train)
            preds = get_predictions(model_svr)
            description= "deg={} eps={}".format(degree, epsilon)
            all_preds[description] = preds
            mse, r2 = get_metrics(description, preds)
            all_mse = all_mse.append(mse, ignore_index=True)
            all_r2 = all_r2.append(r2, ignore_index=True)

if kernel == "rbf":
    for gamma in gammas:
        for epsilon in epsilons:
            model_svr = SVR(kernel=kernel, epsilon=epsilon, gamma=gamma, coef0=1)
            model_svr.fit(X_train, y_train)
            preds = get_predictions(model_svr)
            description= "gamma={} eps={}".format(gamma, epsilon)
            all_preds[description] = preds
            mse, r2 = get_metrics(description, preds)
            all_mse = all_mse.append(mse, ignore_index=True)
            all_r2 = all_r2.append(r2, ignore_index=True)

if kernel == "sigmoid":
    for epsilon in epsilons:
        model_svr = SVR(kernel=kernel, epsilon=epsilon, gamma=gamma, coef0=1)
        model_svr.fit(X_train, y_train)
        preds = get_predictions(model_svr)
        description= "sigmoid, eps={}".format(gamma, epsilon)
        all_preds[description] = preds
        mse, r2 = get_metrics(description, preds)
        all_mse = all_mse.append(mse, ignore_index=True)
        all_r2 = all_r2.append(r2, ignore_index=True)


# Calculate averages of metrics
all_mse['modelAverage'] = all_mse.drop("model", axis=1).mean(axis=1)
all_r2['modelAverage'] = all_r2.drop("model", axis=1).mean(axis=1)

#%%
# Create a new figure and axis
import math

square_root = math.ceil(math.sqrt(len(all_preds.keys())))

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
plot_source = "cv"

sources = list(all_preds.keys())
for i, ax in enumerate(axes.flat):
    if i >= len(sources):
        continue
    source = sources[i]
    ax.scatter(targets[plot_source], all_preds[source][plot_source], label="prediction")
    ref = np.linspace(min(targets[plot_source]), max(targets[plot_source]))
    ax.plot(ref, ref, "k--")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title(source)
    ax.legend()

plt.tight_layout()
plt.show()
print(square_root)
# %% Plot results
def plot_results(metric, description):
    plt.plot(
        range(metric.shape[0]),
        metric["train"],
        linewidth=4,
        markersize=12,
        c="b",
        marker="o",
        label="train",
    )
    plt.plot(
        range(metric.shape[0]),
        metric["test"],
        linewidth=4,
        markersize=12,
        c="r",
        marker="o",
        label="test",
    )
    plt.plot(
        range(metric.shape[0]),
        metric["cv"],
        linewidth=4,
        markersize=12,
        c="g",
        marker="o",
        label="cv",
    )
    plt.xlabel("index value")
    plt.ylabel(description)
    plt.legend()
    plt.show()


plot_results(all_mse, "RMSE")
plot_results(all_r2, "R2")

# %%
best_model = SVR(kernel="poly", epsilon=7, degree=7, coef0=1)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_cv)
print("mse:", mean_squared_error(y_cv, predictions))
print("prediction sum:", predictions.sum())
print("actual sum:", y_cv.sum())
r2_score(y_cv, predictions)
# %%
# %%
import pickle

# open a file to save PCA Matrix
file = open('SVM_ep7_d7.pickle', 'wb')

# dump information to that file
pickle.dump(best_model, file)

# close the file
file.close()
