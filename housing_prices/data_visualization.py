import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import data_frame_imputer as dfi

# Always display all the columns
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)

# features which need to converted to numerical categories
TEXTUAL = {'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
           'Neighborhood','Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
           'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
           'BsmtFinType2','Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence','MiscFeature',
           'MoSold', 'SaleType', 'SaleCondition'}

CATEGORICAL = {'MSSubClass', 'OverallQual', 'OverallCond'}


NUMERIC = {'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF','2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal'}

DATES = {'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'}


def test_df(df):
    # Example on how to count null values in Lot Frontage Column
    # print(df.columns)
    print(df['LandContour'])
    print(df['BldgType'])
    print(df['LotArea'])
    print(df['GarageCars'])


def columns_with_null(df):
    return df.columns[df.isnull().any()].tolist()


def fill_null_columns(df):
    null_columns = columns_with_null(df)
    data_frame_imputer = dfi.DataFrameImputer(null_columns)
    return data_frame_imputer.fit_transform(df)


def low_variability_column(all_counts, column, threshold=0.9):
    """
    Figure out which columns have very low variability and can hence not be considered when selecting features
    :param all_counts:
    :param column:
    :param threshold:
    :return:
    """
    total_count = all_counts.sum()
    return (all_counts.nlargest(1).values[0]/total_count) >= threshold

# fixme:remove for loop
def find_droppable_columns(df):
    # we have no use of the id column and hence we init dropped_columns with id
    dropped_columns = {'Id'}
    for column in df:
        #plt.title(column)
        #value_counts = housing_df[column].value_counts()
        #print(value_counts[:10])
        #value_counts[:10].plot(kind='bar')
        #plt.show(block=True)
        all_counts = df[column].value_counts()
        if low_variability_column(all_counts, column):
            dropped_columns.add(column)

    print("Number of columns which can be dropped is %d" % len(dropped_columns))
    return dropped_columns


def convert_to_categorical(df, columns):
    for column in columns:
        df[column] = df[column].astype('category')
    return df


def preprocess_data(df, dropped_columns):
    df = df.drop(list(dropped_columns), axis=1)
    textual_columns = list(TEXTUAL - dropped_columns)
    numeric_columns = list(NUMERIC - dropped_columns)
    categorical_columns = list(CATEGORICAL|TEXTUAL - dropped_columns)
    # first convert textual columns to categorical datatype
    df = convert_to_categorical(df, textual_columns)
    # now convert those categorical data types to numeric representations
    df[textual_columns] = df[textual_columns].apply(lambda x: x.cat.codes)
    # scale numeric data
    min_max_scaler = preprocessing.MinMaxScaler()
    df[numeric_columns] = min_max_scaler.fit_transform(df[numeric_columns])
    # one hot encode categorical data
    one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
    #df[categorical_columns] = one_hot_encoder.fit_transform(df[categorical_columns])
    # for c in categorical_columns:
    #     print(c)
    #     print(df[c])
    #     df[c] = one_hot_encoder.fit_transform(df[c])
    # fixme: take care of date columns
    # fixme: take care textual categorical columns (use pandas get_dummies)
    return df

if __name__ == '__main__':
    # 1. read data from csv
    housing_df = pd.read_csv('train.csv')
    # 2. Fill in missing values
    housing_df = fill_null_columns(housing_df)
    # 3. Find columns which can be dropped
    dropped_columns = find_droppable_columns(housing_df)
    #test_df(housing_df)
    # 4. Preprocess Data
    housing_df = preprocess_data(housing_df, dropped_columns)
    #test_df(housing_df)
