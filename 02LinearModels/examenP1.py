import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# %%
# train = pd.read_csv("Data/Audit_train.csv", index_col=0)
# test = pd.read_csv("Data/Audit_test.csv", index_col=0)
# unknown = pd.read_csv("Data/Audit_unknown.csv", index_col=0)
train = pd.read_csv("Data/train.csv", index_col=0)
test = pd.read_csv("Data/test.csv", index_col=0)
unknown = pd.read_csv("Data/unknown.csv", index_col=0)


# %%
X_train = train.drop(columns=["Audit_Risk", "Detection_Risk", "PROB"])
y_train = train["Audit_Risk"]
X_train.shape, y_train.shape

# %%
X_test = test.drop(columns=["Audit_Risk", "Detection_Risk", "PROB"])
y_test = test["Audit_Risk"]

# %%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def get_model_metrics(actual, predicted):
    print("RMSE: %0.2f" % (mean_squared_error(actual, predicted)))
    print("r2_score:  %0.2f" % r2_score(actual, predicted))


# %%
def get_outliers(df_var, distance=1.5):
    lower = 0
    std = df_var.std()
    outliers = np.where(
        df_var > df_var.mean() + std * distance,
        True,
        np.where(df_var < lower, True, False),
    )
    return df_var.loc[outliers]


def get_all_outliers_indexes(df, print_percentages=False, distance=1.5):
    outliers_set = set()
    for col in df.columns:
        outliers = get_outliers(df[col], distance=distance)
        outliers_amount = outliers.shape[0]
        if print_percentages:
            print(
                col,
                "-> ",
                outliers_amount,
                " outliers represent: ",
                round(outliers_amount / df[col].shape[0] * 100, 2),
                "%",
            )
        outliers_set.update(outliers.index)
    if print_percentages:
        print(
            "TOTAL outliers represent: ",
            round(len(outliers_set) / df[col].shape[0] * 100, 2),
            "%",
        )
    return outliers_set


# %%
outliers = get_all_outliers_indexes(X_train, distance=4)
len(outliers)

# %%
X_train_outliers = X_train.drop(outliers)
y_train_outliers = y_train.drop(outliers)
train_mean = X_train_outliers.mean()
train_std = X_train_outliers.std()
X_train_outliers = (X_train_outliers - train_mean) / train_std
X_test_outliers = (X_test - train_mean) / train_std
train_mean, train_std


#%%
# No Outliers MODEL
outliers_model = LinearRegression()
outliers_model.fit(X_train_outliers, y_train_outliers)
print("Training:")
predicted = outliers_model.predict(X_train_outliers)
get_model_metrics(y_train_outliers, predicted)

print("Testing:")
predicted = outliers_model.predict(X_test_outliers)
get_model_metrics(y_test, predicted)


# Training:
# RMSE: 5.20
# r2_score:  0.91
# Testing:
# RMSE: 81.99
# r2_score:  0.89

# Intercept: 3.374968719576715

# Coefficients:
# +--+-------------+
# |  |0            |
# +--+-------------+
# |0 |1.638521e-01 |
# |1 |7.687972e+00 |
# |2 |-1.463487e-01|
# |3 |3.411570e+00 |
# |4 |7.585943e+01 |
# |5 |-3.520338e-01|
# |6 |1.991771e+00 |
# |7 |-8.087195e+01|
# |8 |7.364799e-02 |
# |9 |7.364799e-02 |
# |10|7.364799e-02 |
# |11|2.514837e+00 |
# |12|3.686312e-01 |
# |13|-1.882896e-01|
# |14|4.888868e-01 |
# |15|-3.333204e-27|
# |16|4.888868e-01 |
# |17|7.067463e-01 |
# |18|7.067463e-01 |
# |19|7.067463e-01 |
# |20|-0.064465    |
# |21|1.179049     |
# |22|0.677163     |
# +--+-------------+


# %%
unknown_trans = pd.DataFrame.copy(unknown.drop(columns=["Detection_Risk", "PROB"]))
unknown_trans = (unknown_trans - train_mean) / train_std
predicted = outliers_model.predict(unknown_trans)
unknown["Audit_Risk"] = predicted

unknown.to_csv("Audit_unknown_predicted.csv")

# %%
