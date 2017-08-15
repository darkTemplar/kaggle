from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
import data_frame_imputer as dfi
import datetime

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
    print(df.columns)


def columns_with_null(df):
    return df.columns[df.isnull().any()].tolist()


def fill_null_columns(df):
    null_columns = columns_with_null(df)
    data_frame_imputer = dfi.DataFrameImputer(null_columns)
    return data_frame_imputer.fit_transform(df)


def low_variability_column(all_counts, column, threshold=0.8):
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
    d = defaultdict(preprocessing.LabelEncoder)
    return df[columns].apply(lambda x: d[x.name].fit_transform(x))


def encode_categorical_data(df, columns):
    return pd.get_dummies(df, columns=columns)


def encode_date_data(df, columns):
    now = datetime.datetime.now()
    return df[columns].apply(lambda x: now.year - x, axis=1)


def preprocess_data(df, dropped_columns):
    df = df.drop(list(dropped_columns), axis=1)
    # take care of date columns
    df[list(DATES)] = encode_date_data(df, list(DATES))
    textual_columns = list((TEXTUAL|CATEGORICAL) - dropped_columns)
    numeric_columns = list((NUMERIC|DATES) - dropped_columns)
    #categorical_columns = list(CATEGORICAL|TEXTUAL - dropped_columns)
    # first convert textual columns to categorical
    #df[textual_columns] = convert_to_categorical(df, textual_columns)
    # one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
    # df[categorical_columns] = one_hot_encoder.fit_transform(df[categorical_columns])
    df = encode_categorical_data(df, textual_columns)
    # scale numeric data
    min_max_scaler = preprocessing.MinMaxScaler()
    df[numeric_columns] = min_max_scaler.fit_transform(df[numeric_columns])
    return df


def extract_sales_price(df):
    prices = df['SalePrice']
    df = df.drop(['SalePrice'], axis=1)
    return df, prices


def kfold_cross_validation(X, y, model='linear', params={}, splits=5):
    if model == 'svr':
        model = get_svm(params.get('kernel', 'linear'))
    elif model == 'ridge':
        model = get_ridge_regression(params.get('alpha', 0.1))
    elif model == 'gradient_boosting':
        model = get_gradient_boosting_regression(params)
    else:
        model = get_least_squares_regression()

    kf = KFold(n_splits=splits, shuffle=True)
    for k, (train, test) in enumerate(kf.split(X, y)):
        model.fit(X[train], y[train])
        print("[fold {0}] score: {1:.5f}".format(k, model.score(X[test], y[test])))


def grid_search(X, y, model='svr'):
    if model == 'svr':
        C_values = [1, 10, 100, 1000, 10000]
        gammas = [0.0001, 0.001, 0.01, 0.1]
        param_grid = {'C': C_values, 'gamma': gammas}
        grid_search = GridSearchCV(SVR(kernel='rbf', cache_size=500), param_grid, cv=5)
        #param_grid = {'C': C_values}
        #grid_search = GridSearchCV(SVR(kernel='linear', cache_size=500), param_grid, cv=5)
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        print(grid_search.best_score_)
        return grid_search.best_params_
    elif model == 'ridge':
        reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        reg.fit(X, y)
        print(reg.alpha_)
        return reg.alpha_
    elif model == 'gradient_boosting':
        return 0


def get_svm(kernel='linear'):
    if kernel == 'rbf':
        svr = SVR(kernel='rbf', C=1e4, gamma=0.01, cache_size=500)
    elif kernel == 'poly':
        svr = SVR(kernel='poly', C=1e3, degree=2, cache_size=500)
    else:
        svr = SVR(kernel='linear', C=1e3, cache_size=500)

    return svr


def get_least_squares_regression():
    lr = LinearRegression()
    return lr


def get_ridge_regression(alpha):
    ridge = Ridge(alpha=alpha)
    return ridge


def get_gradient_boosting_regression(params):
    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
    clf = GradientBoostingRegressor(**params)
    return clf


def prepare_submission_csv():
    pass

if __name__ == '__main__':
    # 1. read data from csv
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # 2. Fill in missing values
    train_df = fill_null_columns(train_df)
    test_df = fill_null_columns(test_df)
    # 3. Find columns which can be dropped
    dropped_columns = find_droppable_columns(train_df)
    #test_df(housing_df)
    # 4. Preprocess Data
    train_df = preprocess_data(train_df, dropped_columns)
    test_df = preprocess_data(test_df, dropped_columns)
    #test_df(housing_df)
    # 5. extract sales price from the data and remove it from dataframe
    train_df, prices = extract_sales_price(train_df)
    # convert data to numpy arrays in prep for being used in learning algos
    X, y = train_df.as_matrix(), prices.values
    #print(X.shape)
    #print(y.shape)

    # use the grid search to determine best params for the different regression algos
    #grid_search(X, y, 'ridge')
    # once we have the best params we use the k fold cross validation on our training set
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
    kfold_cross_validation(X, y, 'gradient_boosting', params)
