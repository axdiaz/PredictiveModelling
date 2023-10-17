import pandas as pd
#%%
df1 = pd.read_csv("data19 2.csv", header=None)
df1.columns = ["name", "size", "grants", "revokes", "subcalls","gvs" ,"mds", "loads"];
#%%
df1
#%%
df2 = pd.read_csv("times19.csv", header=None)
df2.columns = ["name", "time"]
#%%
df2
#%%
df2[df2[0] == "cdcore_mig"][1]
#%%
df = pd.DataFrame()
i = 0
datapoint = {}
for row in df2[0]:
    print(row)
#%%
df = df2.merge(df1, on='name', how='inner')
#%%
df
#%%
df.drop("name", axis=1, inplace=True)
#%%
df.drop("name", axis=1).to_csv("scr_data.csv")
#%%
df
#%%
df.corr()
#%%
df.hist()
#%%
df["time"].desc()
#%%
df ** 2
#%%
df.hist()
#%%
df_low = df[df["time"] < 10]
#%%
df_low.hist()
#%%
df_low.corr()
#%%
df_high = df[df["time"] > 10]
#%%
df_high.corr()
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,r2_score)

#%%
linear_reg = LinearRegression()
linear_reg.fit(df_low.drop("time", axis=1), df_low["time"])
#%%
y_predict = linear_reg.predict(df_low.drop("time", axis=1))
#%%
r2_score(df_low["time"], y_predict)
#%%
linear_reg = LinearRegression()
linear_reg.fit(df_high.drop("time", axis=1), df_high["time"])
#%%
y_predict = linear_reg.predict(df_high.drop("time", axis=1))
#%%
r2_score(df_high["time"], y_predict)
#%%
df_high
#%%
#%%
linear_reg = LinearRegression()
linear_reg.fit(df.drop("time", axis=1), df["time"])
#%%
y_predict = linear_reg.predict(df.drop("time", axis=1))
#%%
r2_score(df["time"], y_predict)

#%%
