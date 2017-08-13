import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import data_frame_imputer as dfi

# Always display all the columns
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)
housing_df = pd.read_csv('train.csv')

CATEGORICAL = {'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
               'LandSlope', 'Neighborhood', 'Condition1', 'Condition2, ''BldgType', 'HouseStyle', 'OverallQual',
               'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
               'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
               'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
               'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
               'MiscFeature', 'MoSold', 'SaleType', 'SaleCondition'}

NUMERIC = {'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
           '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
           'ScreenPorch', 'PoolArea', 'MiscVal'}

DATES = {'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'}


def test_df():
    global housing_df
    # Example on how to count null values in Lot Frontage Column
    # print(housing_df.columns)
    print(housing_df['MoSold'])
    print(housing_df['GarageType'])
    print(housing_df['LotArea'])
    print(housing_df['GarageCars'])


def columns_with_null():
    global housing_df
    return housing_df.columns[housing_df.isnull().any()].tolist()


def fill_null_columns(null_columns):
    global housing_df
    data_frame_imputer = dfi.DataFrameImputer(null_columns)
    housing_df = data_frame_imputer.fit_transform(housing_df)


def low_variability_column(column, threshold=0.9):
    """
    Figure out which columns have very low variability and can hence not be considered when selecting features
    :param column:
    :param threshold:
    :return:
    """
    global housing_df
    all_counts = housing_df[column].value_counts()
    total_count = all_counts.sum()
    return (all_counts.nlargest(1).values[0]/total_count) >= threshold


def find_droppable_columns():
    # we have no use of the id column and hence we init dropped_columns with id
    dropped_columns = {'Id'}
    for column in housing_df:
        plt.title(column)
        value_counts = housing_df[column].value_counts()
        #print(value_counts[:10])
        #value_counts[:10].plot(kind='bar')
        #plt.show(block=True)
        if low_variability_column(column):
            dropped_columns.add(column)

    print("Number of columns which can be dropped is %d" % len(dropped_columns))
    print(dropped_columns)
    return dropped_columns


def preprocess_data():
    global housing_df
    dropped_columns = find_droppable_columns()
    categorical_columns = list(CATEGORICAL - dropped_columns)
    numeric_columns = list(NUMERIC - dropped_columns)
    # scale numeric data
    min_max_scaler = preprocessing.MinMaxScaler()
    housing_df[numeric_columns] = min_max_scaler.fit_transform(housing_df[numeric_columns])
    # one hot encode categorical data
    one_hot_encoder = preprocessing.OneHotEncoder()
    housing_df[categorical_columns] = one_hot_encoder.fit_transform(housing_df[categorical_columns])
    # fixme: take care of date columns

if __name__ == '__main__':
    #test_df()
    #preprocess_data()
    #test_df()
    null_columns = columns_with_null()
    print(null_columns)
    fill_null_columns(null_columns)
    print(columns_with_null())