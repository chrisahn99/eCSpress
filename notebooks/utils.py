from itertools import combinations
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Tuple


def get_rmse(y_pred: pd.DataFrame, y_true: pd.DataFrame) -> float:
    """ Calculates RMSE """
    MSE = mean_squared_error(y_true, y_pred)
    return math.sqrt(MSE)

def plot_series(df: pd.DataFrame, title: str) -> None:
    """ Plots a series """
    plt.figure(figsize=(17, 2))
    plt.plot(df)    
    plt.xticks(rotation=45)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_sortie_acf(y_acf: np.ndarray, y_len: int, pacf: bool =False) -> None:
    """ Plots an ACF """
    if pacf:
        y_acf = y_acf[1:]
    plt.figure(figsize=(16, 2))
    plt.bar(range(len(y_acf)), y_acf, width = 0.1)
    plt.xlabel('lag')
    plt.ylabel('ACF')
    plt.title('ACF')
    plt.axhline(y=0, color='black')
    plt.axhline(y=-1.96/np.sqrt(y_len), color='b', linestyle='--', linewidth=0.8)
    plt.axhline(y=1.96/np.sqrt(y_len), color='b', linestyle='--', linewidth=0.8)
    plt.ylim(-1, 1)
    plt.show()

def timeseries_train_test_split(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Perform train-test split with respect to time series structure
    """
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test

def add_lag_features(df: pd.DataFrame, variable_name: str) -> Tuple[pd.DataFrame, str]:
    """ Function to add the corresponding lag feature of 7 days.
        - Copies the dataframe
        - Adds the feature
        - Drops missing values
        - Resets the index
    """
    df_bis = df.copy()

    lag_feat_name = 'Lag_7_' + variable_name[0]
    df_bis[lag_feat_name] = df_bis[variable_name].shift(7*24)
    # # Selecting features
    # df_bis = df_bis[features_tot + [lag_feat_name] + variables]
    # Drop missing values
    df_bis.dropna(inplace=True)
    # Reset index
    df_bis = df_bis.reset_index(drop=True)
    return df_bis, lag_feat_name

def process_subset(X, y, feature_set, plot=False):
    """ Function to do a Linear regression
    Input:
    - X: feature dataset
    - y: target value
    - feature_set: set of feature to consider
    Returns:
    - dict of model, RMSE and feature_set
    """
    X = X[list(feature_set)]

    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y)

    model = LinearRegression()
    regr = model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    RMSE = round(get_rmse(y_pred, y_test), 1)

    if plot:
        plot_params = dict(
            color="0.75",
            style=".-",
            markeredgecolor="0.25",
            markerfacecolor="0.25",
            legend=False,
        )

        ax = y_test.plot(**plot_params)
        ax = y_pred.plot(ax=ax, linewidth=3)
        ax.set_title('Plot')
    return {"model": regr, "RMSE": RMSE, "Features": feature_set}


def forward(X, y, features, features_tot):
    """ Forward subset selection """
    # Pull out features we still need to process
    remaining_features = [d for d in features_tot if d not in features]
    results = []

    for d in (remaining_features):
        result = process_subset(X=X, y=y, feature_set=features+[d])
        results.append(result)

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the lowest RMSE
    best_model = models.loc[models['RMSE'].argmin()]

    # Return the best model, along with some other useful information about the model
    return best_model


def backward(X, y, features):
    """ Backward subset selection """
    results = []

    for combo in tqdm(combinations(features, len(features)-1)):
        results.append(process_subset(X=X, y=y, feature_set=combo))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest RSS
    best_model = models.loc[models['RMSE'].argmin()]

    # Return the best model, along with some other useful information about the model
    return best_model


def feature_selection(df: pd.DataFrame, variable: str, all_features: list) -> None:
    """ Process to do foreward and backward selection and plot evolution of RMSE
    Input:
    - df: entire DataFrame
    - variable: string of target name
    - all_features: list of all possible features
    """
    X = df[all_features]
    y = df[variable]

    # Backward
    print("BACKWARD")
    models_bwd = pd.DataFrame(columns=["RMSE", "model", "Features"], index = range(1, len(all_features)))
    features = all_features.copy()
    while (len(features) > 1):
        models_bwd.loc[len(features)-1] = backward(X, y, features)
        features = models_bwd.loc[len(features)-1]["Features"]

    # Forward
    print("FOREWARD")
    models_fwd = pd.DataFrame(columns=["RMSE", "model", "Features"])
    features = []
    for i in tqdm(range(1, len(all_features)+1)):
        models_fwd.loc[i] = forward(X, y, features, all_features)
        features = models_fwd.loc[i]["Features"]

    # Plot RMSE values
    rmse_bwd = []
    for line in models_bwd.index:
        rmse_bwd += [models_bwd.loc[line]["RMSE"]]
    plt.plot(rmse_bwd)
    plt.title("Backward")
    plt.xticks(np.arange(len(rmse_bwd)))
    plt.show()

    rmse_fwd = []
    for line in models_fwd.index:
        rmse_fwd += [models_fwd.loc[line]["RMSE"]]
    plt.plot(rmse_fwd)
    plt.title("Foreward")
    plt.xticks(np.arange(len(rmse_fwd)))
    plt.show()

    return models_fwd, models_bwd
