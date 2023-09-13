# -*- coding: utf-8 -*-

# Import libraries to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% IMPORT THE DATA SET
data = pd.read_csv("Data/glass.data", header=None)
names = [
    "ID",
    "Refractive_index",
    "Na",
    "Mg",
    "Al",
    "Si",
    "K",
    "Ca",
    "Ba",
    "Fe",
    "Glass_type",
]
data.columns = names


# %% DATA QUALITY REPORT
def dqr(data):
    # List of database variables
    cols = pd.DataFrame(
        list(data.columns.values), index=list(data.columns.values)
    )
    # List of data types
    dtyp = pd.DataFrame(data.dtypes, columns=["Type"])
    # List of missing data
    misval = pd.DataFrame(data.isnull().sum(), columns=["Missing_values"])
    # List of present data
    presval = pd.DataFrame(data.count(), columns=["Present_values"])
    # List of unique values
    unival = pd.DataFrame(columns=["Unique_values"])
    # List of min values
    minval = pd.DataFrame(columns=["Min_value"])
    # List of max values
    maxval = pd.DataFrame(columns=["Max_value"])
    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        if not pd.api.types.is_numeric_dtype(data.dtypes[col]):
            minval.loc[col] = [None]
            maxval.loc[col] = [None]
        else:
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
    # Join the tables and return the result
    return (
        cols.join(dtyp)
        .join(misval)
        .join(presval)
        .join(unival)
        .join(minval)
        .join(maxval)
    )


# %% Obtaining the data quality report
report = dqr(data)

# %% View one of the variables
fig = plt.figure(figsize=(10, 5))
plt.scatter(data.ID, data.Refractive_index)
plt.xlabel("ID"), plt.ylabel("Refractive index")
plt.grid()
plt.show()
# fig.savefig('../figures/P1_fig/F1.png')

# %% View one of the variables
fig = plt.figure(figsize=(5, 4))
plt.scatter(data.ID, data.Na)
plt.xlabel("ID"), plt.ylabel("Sodio (Na)")
plt.grid()
plt.show()
# fig.savefig('../figures/P1_fig/F2.png')

# %%  View one of the variables
fig = plt.figure(figsize=(5, 4))
plt.scatter(data.Refractive_index, data.Na)
plt.xlabel("Refractive index"), plt.ylabel("Sodio (Na)")
# plt.axis('square')
plt.grid()
plt.show()
# fig.savefig('../figures/P1_fig/F3.png')

# %% SCALING OF VARIABLES BY NORMALIZATION
data["Refractive_index_scale"] = (
    data.Refractive_index - data.Refractive_index.mean()
) / data.Refractive_index.std()
data["Na_scale"] = (data.Na - data.Na.mean()) / data.Na.std()

### Scaling through scikit-learn
# from sklearn import preprocessing
# data['Refractive_index_scale'] = preprocessing.scale(data.Refractive_index)


# %% Display the new variable
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(data.ID, data.Na)
plt.xlabel("ID"), plt.ylabel("Sodio (Na)")
plt.title("Original")
plt.grid()
plt.subplot(1, 2, 2)
plt.scatter(data.ID, data.Na_scale)
plt.xlabel("ID"), plt.ylabel("Sodio (Na) Rescaled")
plt.title("Rescaled")
plt.grid()
fig.tight_layout()
plt.show()
# fig.savefig('../figures/P1_fig/F4.png')

# %% Display the new variable
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(data.Refractive_index, data.Na)
plt.xlabel("Refractive index"), plt.ylabel("Sodio (Na)")
plt.axis("square")
plt.title("Original")
plt.grid()
plt.subplot(1, 2, 2)
plt.scatter(data.Refractive_index_scale, data.Na_scale)
plt.xlabel("Refractive index"), plt.ylabel("Sodio (Na)")
plt.axis("square")
plt.title("Rescaled")
plt.grid()
fig.tight_layout()
plt.show()
# fig.savefig('../figures/P1_fig/F5.png')


# %% ASYMMETRY IN THE VARIABLES
fig = plt.figure()
plt.hist(data.Refractive_index, bins=30)
plt.xlabel("Refractive_index"), plt.ylabel("Frequency")
plt.vlines(data.Refractive_index.mean(), 0, 50, "r")
plt.show()
# fig.savefig('../figures/P1_fig/F6.png')


# %% Empirical criterion to consider that the data may have asymmetry
ratio = data.max() / data.min()

# %% Calculation of skewness
v = np.sum(np.power(data - data.mean(axis=0), 2)) / (data.shape[0] - 1)
skewness = np.sum(np.power(data - data.mean(axis=0), 3)) / (
    (data.shape[0] - 1) * np.power(v, 3 / 2)
)

## Calculation of skewness with pandas
# skewness = data.skew()

## Calculation of skewness with scipy
# from scipy import stats
# skewness = stats.skew(data)

# %% Skewness verification by means of histograms
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.hist(data.Refractive_index)
plt.xlabel("Refractive_index"), plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(data.Mg)
plt.xlabel("Magnesium (Mg)"), plt.ylabel("Frequency")
fig.tight_layout()
plt.show()

# %% Transformation to limit skewness
data["Refractive_index_no_skewness"] = np.sqrt(data.Refractive_index)
# data['Refractive_index_no_skewness'] = np.log(data.Refractive_index)
# data['Refractive_index_no_skewness'] = 1/data.Refractive_index


### BoxCox transformation using scipy
# from scipy import stats
# data['Refractive_index_no_skewness'] = stats.boxcox(data.Refractive_index,lmbda=-5)

# data['Refractive_index_no_skewness'],lamb = stats.boxcox(data.Refractive_index)


# %% Skewness check
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.hist(data.Refractive_index)
plt.xlabel("Refractive_index"), plt.ylabel("Frequency")
plt.title("Skewness: %0.3f" % data["Refractive_index"].skew())
plt.subplot(1, 2, 2)
plt.hist(data.Refractive_index_no_skewness)
plt.xlabel("Refractive_index_no_skewness"), plt.ylabel("Frequency")
plt.title("Skewness: %0.3f" % data["Refractive_index_no_skewness"].skew())
fig.tight_layout()
plt.show()
# fig.savefig('../figures/P1_fig/F7.png')


# %%  IDENTIFICATION OF ATYPICAL VALUES
import seaborn as sns

# A boxplot is not affected by scaling
sns.boxplot(y=data["Refractive_index"])
# sns.boxplot(y=data['Refractive_index_scale'])


# %% Function to determine the outliers
def find_boundaries(df_var, distance=1.5):
    IQR = df_var.quantile(0.75) - df_var.quantile(0.25)
    lower = df_var.quantile(0.25) - IQR * distance
    upper = df_var.quantile(0.75) + IQR * distance
    return lower, upper


lmin, lmax = find_boundaries(data["Refractive_index"])
outliers = np.where(
    data["Refractive_index"] > lmax,
    True,
    np.where(data["Refractive_index"] < lmin, True, False),
)
outliers_df = data.loc[outliers, "Refractive_index"]

# %% Spatial sign transformation to mitigate outliers
tmp = data[["Refractive_index_scale", "Na_scale"]]
# modulo = np.sqrt(np.sum(tmp*tmp,axis=1))
# tmp['Refractive_index_scale'] = tmp['Refractive_index_scale']/modulo
# tmp['Na_scale'] = tmp['Na_scale']/modulo
# plt.scatter(tmp.Refractive_index_scale,tmp.Na_scale)

# Using scikit-learn
from sklearn import preprocessing

tmp = preprocessing.normalize(np.array(tmp), norm="l2")
fig = plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
plt.scatter(data["Refractive_index_scale"][outliers], data["Na_scale"][outliers], c="r")
plt.scatter(data["Refractive_index_scale"][~outliers], data["Na_scale"][~outliers])
plt.xlabel("Refractive_index_scale"), plt.ylabel("Na_scale")
plt.subplot(1, 2, 2)
plt.scatter(tmp[outliers, 0], tmp[outliers, 1], c="r")
plt.scatter(tmp[~outliers, 0], tmp[~outliers, 1])
plt.xlabel("Refractive_index_scale"), plt.ylabel("Na_scale")
plt.tight_layout()
plt.show()
# fig.savefig('../figures/P1_fig/F8.png')


# %% APPLICATION OF PCA TO DATA
from sklearn.decomposition import PCA
from sklearn import preprocessing

tmp = preprocessing.scale(data.iloc[:, 1:10])
pca = PCA()
# pca = PCA(n_components=3)
pca.fit(tmp)
data_pca = pca.transform(tmp)
components = pca.components_


# %% TREATMENT OF MISSING DATA
# % Import the table
datamovie = pd.read_excel("..\Data\movietest\Test de películas (anónimo)(1-12).xlsx")
# %% selection of valid columns
csel = np.arange(19, 247, 3)
cnames = list(datamovie.columns.values[csel])
datan = datamovie[cnames]

# %% Delete all records with null data
datan_clean = datan.dropna()
miss_val_data = datan.isnull().mean().sort_values(ascending=True)

# %% Imputation by mean or median
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="constant", fill_value=0)
# imputer = SimpleImputer(strategy='constant', fill_value='Missing')
# imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy='median')
# imputer = SimpleImputer(strategy='most_frequent')

imputer.fit(datan)
imputer.statistics_  # review the values to be replaced by
datan_clean = imputer.transform(datan)

# %% Iterative imputation, valid for scikit-learn 0.23.2
# from sklearn.linear_model import BayesianRidge
# from sklearn.impute import IterativeImputer
# from sklearn.neighbors import KNeighborsRegressor
#
# imputer = IterativeImputer(estimator = BayesianRidge(),max_iter=10, random_state=0)
# imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5),max_iter=10,random_state=0)
# imputer.fit(datan)
# data_clean = imputer.transform(datan)

# %% VARIANCE CRITERIA FOR ELIMINATION OF VARIABLES
variances = pd.DataFrame(datan.var().sort_values(), columns=["Variance"])
fig = plt.figure(figsize=(10, 8))
plt.bar(np.arange(len(variances)), variances.Variance)
plt.ylabel("Variance")
plt.xticks(np.arange(len(variances)), variances.index, rotation=90)
plt.tight_layout()
plt.show()
# fig.savefig('../figures/P1_fig/F9.png')

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=1)
sel.fit_transform(datan)


# %% Correlation analysis for elimination of variables
import matplotlib.pyplot as plt

subdata = data.iloc[:, 1:10]
correlations = np.corrcoef(subdata, rowvar=False)
fig = plt.figure()
plt.imshow(correlations)
plt.xticks(np.arange(9), np.arange(9))
plt.yticks(np.arange(9), np.arange(9))
plt.colorbar()
plt.show()
# fig.savefig('../figures/P1_fig/F10.png')

# %% Hierarchical clustering application
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(subdata.T, metric="correlation", method="complete")

d = dendrogram(Z)
plt.show()
correlaciones_clust = np.corrcoef(subdata.iloc[:, d["leaves"]], rowvar=False)
fig = plt.figure()
plt.imshow(correlaciones_clust)
plt.xticks(np.arange(9), d["leaves"])
plt.yticks(np.arange(9), d["leaves"])
plt.colorbar()
plt.show()
# fig.savefig('../figures/P1_fig/F11.png')


# %% Use of Seaborn library
import seaborn as sns

# Option 1. "all-vs-all", individual histogram
sns.pairplot(data)

# Option 2. Separation of distributions based on output variable
# sns.pairplot(data,hue='Glass_type')

# Option 3. Select variables to plot "all-vs-all"
# sns.pairplot(data,vars=['Refractive_index','Na','Mg'])

# Option 4. Selection of variables of interest
# fig = sns.pairplot(data,x_vars=['Na','Mg'],
#              y_vars=['Refractive_index'])

# %% MAKING USE OF PANDAS PROFILING



import pandas_profiling

# report = data.profile_report()
report = pandas_profiling.ProfileReport(data)
report.to_file(output_file="Titanic data profiling.html")

#%%
