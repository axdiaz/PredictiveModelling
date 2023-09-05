import pandas as pd
import numpy as np


def data_quality_report(data):
    # List of database variables
    cols = pd.DataFrame(
        list(data.columns.values), columns=["Names"], index=list(data.columns.values)
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
    minval = pd.DataFrame(columns=["Min value"])
    # List of max values
    maxval = pd.DataFrame(columns=["Max_value"])
    for col in list(data.columns.values):
        unival.loc[col] = [data[col].nunique()]
        if pd.api.types.is_numeric_dtype(data.dtypes[col]):
            minval.loc[col] = [data[col].min()]
            maxval.loc[col] = [data[col].max()]
        else:
            pass
    # Join the tables and return the result
    return (
        cols.join(dtyp)
        .join(misval)
        .join(presval)
        .join(unival)
        .join(minval)
        .join(maxval)
    )


def get_outliers(df_var, distance=1.5):

    IQR = df_var.quantile(0.75) - df_var.quantile(0.25)
    lower = df_var.quantile(0.25) - IQR * distance
    upper = df_var.quantile(0.75) + IQR * distance
    outliers = np.where(
        df_var > upper,
        True,
        np.where(df_var < lower, True, False),
        )
    return df_var.loc[outliers]


def get_all_outliers_indexes(df, print_percentages=False):
    outliers_set = set()
    for col in df.columns:
        outliers = get_outliers(df[col])
        outliers_amount = outliers.shape[0]
        if print_percentages:
            print(col, "-> ", outliers_amount, " outliers represent: ", round(outliers_amount / df[col].shape[0] * 100, 2), "%")
        outliers_set.update(outliers.index)
    if print_percentages:
        print("TOTAL outliers represent: ", round(len(outliers_set) / df[col].shape[0] * 100, 2), "%")
    return outliers_set
