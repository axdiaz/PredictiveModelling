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
epsilons = [1, 4, 5, 7, 10, 15, 20,30]
degrees = [5]  # For Poly
gammas = [2, 3, 5, 10, 15]  # For rbf (radial)


# Training
cv_size = 0.2

# Data Preprocessing
drop_outliers = True
fix_skewness = True
scale = False
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


# %%


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
            "{}:{} , RMSE=%0.4f, R^2=%0.4f".format(
                description, source, mean_squared_error(y, pred), r2_score(y, pred)
            )
        )
        plt.grid()


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
        "cv": mean_squared_error(y_cv, preds["cv"]),
    }
    r2_row = {
        "model": description,
        "train": r2_score(y_train, preds["train"]),
        "test": r2_score(y_test, preds["test"]),
        "cv": r2_score(y_cv, preds["cv"]),
    }
    return mse_row, r2_row


def get_predictions(model):
    preds = {"train": model.predict(X_train), "test": model.predict(X_test), "cv": model.predict(X_cv)}
    return preds


# Train Base Linear Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
preds = get_predictions(linear_reg)
mse, r2 = get_metrics("linear regression", preds)
all_mse = all_mse.append(mse, ignore_index=True)
all_r2 = all_r2.append(r2, ignore_index=True)
all_preds["linear"] = preds
all_nvectors = []

# Train SVMs
if kernel == "poly":
    for degree in degrees:
        for epsilon in epsilons:
            model_svr = SVR(kernel=kernel, epsilon=epsilon, degree=degree, coef0=1)
            model_svr.fit(X_train, y_train)
            preds = get_predictions(model_svr)
            description = "deg={} eps={}".format(degree, epsilon)
            all_preds[description] = preds
            all_nvectors.append(model_svr.n_support_[0])
            mse, r2 = get_metrics(description, preds)
            all_mse = all_mse.append(mse, ignore_index=True)
            all_r2 = all_r2.append(r2, ignore_index=True)

if kernel == "rbf":
    for gamma in gammas:
        for epsilon in epsilons:
            model_svr = SVR(kernel=kernel, epsilon=epsilon, gamma=gamma, coef0=1)
            model_svr.fit(X_train, y_train)
            preds = get_predictions(model_svr)
            description = "gamma={} eps={}".format(gamma, epsilon)
            all_preds[description] = preds
            mse, r2 = get_metrics(description, preds)
            all_mse = all_mse.append(mse, ignore_index=True)
            all_r2 = all_r2.append(r2, ignore_index=True)

if kernel == "sigmoid":
    for epsilon in epsilons:
        model_svr = SVR(kernel=kernel, epsilon=epsilon, coef0=1)
        model_svr.fit(X_train, y_train)
        preds = get_predictions(model_svr)
        description = "sigmoid, eps={}".format(epsilon)
        all_preds[description] = preds
        mse, r2 = get_metrics(description, preds)
        all_mse = all_mse.append(mse, ignore_index=True)
        all_r2 = all_r2.append(r2, ignore_index=True)


# Calculate averages of metrics
# all_mse['modelAverage'] = all_mse.drop("model", axis=1).mean(axis=1)
# all_r2['modelAverage'] = all_r2.drop("model", axis=1).mean(axis=1)

all_r2["drop_outliers"] = drop_outliers
all_r2["fix_skewness"] = fix_skewness
all_r2["scale"] = scale
all_r2["use_pls"] = use_pls
all_r2.to_csv(results_file, mode="a", header=True)

all_mse["drop_outliers"] = drop_outliers
all_mse["fix_skewness"] = fix_skewness
all_mse["scale"] = scale
all_mse["use_pls"] = use_pls
all_mse.to_csv(mse_results_file, mode="a", header=True)

# %%
# Create a new figure and axis
import math

square_root = math.ceil(math.sqrt(len(all_preds.keys())))

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
plot_source = "test"

sources = list(all_preds.keys())
for i, ax in enumerate(axes.flat):
    if i >= len(sources):
        continue
    source = sources[i]
    ax.scatter(targets[plot_source], all_preds[source][plot_source], label="prediction")
    ref = np.linspace(min(targets[plot_source]), max(targets[plot_source]))
    ax.plot(ref, ref, "k--")
    ax.set_xlabel("REAL y")
    ax.set_ylabel("PREDICTED y")
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
best_model = SVR(kernel="poly", epsilon=10, degree=5, coef0=1)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
print("mse:", mean_squared_error(y_test, predictions))
print("prediction sum:", predictions.sum())
print("actual sum:", y_test.sum())

r2_score(y_test, predictions)

ref = np.linspace(min(y_test), max(y_test))
fig = plt.figure(figsize=(10, 8))
plt.scatter(y_test, predictions)
plt.plot(ref, ref, "k--")
plt.axis("square")
plt.xlabel("y real"), plt.ylabel("y predict")
plt.title(
    "epsilon=10, degree=5, coef0=1, RMSE=%0.4f, R^2=%0.4f"%(
        mean_squared_error(y_test, predictions), r2_score(y_test, predictions)
    )
)
plt.grid()

# %%

# open a file to save PCA Matrix
file = open("SVM_ep10_d5.pickle", "wb")

# dump information to that file
pickle.dump(best_model, file)

# close the file
file.close()

#%%
