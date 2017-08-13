from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


class DataFrameImputer(TransformerMixin):

    def __init__(self, null_columns):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with median of column.

        """
        self.null_columns = null_columns

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in self.null_columns],
            index=self.null_columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

"""
Example Usage:
data = [
    ['a', 1, 2],
    ['b', 1, 1],
    ['b', 2, 2],
    [np.nan, np.nan, np.nan]
]

X = pd.DataFrame(data)
xt = DataFrameImputer().fit_transform(X)"""