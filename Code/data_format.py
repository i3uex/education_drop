import pandas as pd

SHA1 = "SHA1"


def encrypt_column(column, algorithm):
    import hashlib
    if algorithm.upper() == 'SHA1':
        return hashlib.sha1(column.encode()).hexdigest()
    elif algorithm.upper() == 'SHA224':
        return hashlib.sha224(column.encode()).hexdigest()
    elif algorithm.upper() == 'SHA256':
        return hashlib.sha256(column.encode()).hexdigest()
    elif algorithm.upper() == 'SHA384':
        return hashlib.sha384(column.encode()).hexdigest()
    elif algorithm.upper() == 'SHA512':
        return hashlib.sha512(column.encode()).hexdigest()
    else:
        raise Exception("Sorry, algorithm is invalid, please select 'SHA1', 'SHA224', 'SHA256', 'SHA384' or 'SHA512'")


def generate_profile(dataset, name_of_dataset):
    from pandas_profiling import ProfileReport
    prof_titulados = ProfileReport(dataset)
    prof_titulados.to_file('profile_' + name_of_dataset + '.html')


def replace_reg_exp_by_nan(ds, regexp):
    import numpy as np
    return ds.replace(regexp, np.nan, regex=True)


def drop_nan_subset(ds, columns):
    return ds.dropna(subset=columns)


def drop_completely_empty_rows(ds):
    return ds.dropna(axis=0, how='all')


def drop_duplicates_filter_by_column(ds, columns):
    ds.drop_duplicates(subset=columns, keep='last', inplace=True)


def calculate_age(column_date_of_birth, format_of_date):
    from datetime import date
    from datetime import datetime
    import numpy as np
    import math
    if not pd.isna(column_date_of_birth):
        current_date = datetime.strptime((date.today().strftime(format_of_date)), format_of_date)
        date_of_birth = datetime.strptime(column_date_of_birth, format_of_date)
        return math.trunc((current_date - date_of_birth).days / 365)
    else:
        return np.nan


def imputation_func(x, index_column_to_impute, ds, name_of_column_filter="na", column_filter="na"):
    import statistics
    if pd.isna(x.iloc[index_column_to_impute]):
        if name_of_column_filter != "na" and column_filter != "na":
            x.iloc[index_column_to_impute] = \
                statistics.median(ds.loc[ds[name_of_column_filter] ==
                                         column_filter].iloc[:, index_column_to_impute].dropna())
        else:
            x.iloc[index_column_to_impute] = statistics.median(ds.iloc[:, index_column_to_impute].dropna())

    return x.iloc[index_column_to_impute]


def delete_values_greater_than(ds, column_name, maximum):
    return ds.loc[(ds[column_name] <= maximum)]


def delete_values_smaller_than(ds, column_name, minimum):
    return ds.loc[ds[column_name] >= minimum]


def replace_reg_exp_in_column(column, old_reg_exp, new_regexp):
    import re
    import numpy as np
    if not pd.isnull(column):
        return re.sub(old_reg_exp, new_regexp, column).strip()
    else:
        return np.nan


def select_columns_of_dataset(ds, columns):
    return ds[columns]
