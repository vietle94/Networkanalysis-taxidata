#! /usr/bin/env python
"""
This module implements ML usecases (prediction and classifier) where
the ML algorithms make use of graph embeddings.
"""

import json
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.cluster import KMeans

data_path = "./data/"

def kf_predict(X, Y, alpha):
    """
    Run Ridge regressions with cross validations

    Returns results from various runs of K-Fold cross validations
    """
    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model = linear_model.Ridge(alpha=alpha).fit(X_train, y_train)
        y_preds.append(model.predict(X_test))
        y_truths.append(y_test)
    return np.concatenate(y_preds), np.concatenate(y_truths)


def compute_metrics(y_pred, y_test):
    """
    Compute prediction accuracy metrics
    """
    y_pred[y_pred < 0] = 0

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    msle = mean_squared_error(np.log(y_test+1), np.log(y_pred+1))
    return mae, rmse, r2, msle

def predict_crime(emb, target):
    """
    Run a prediction
    """
    maes, rmses, r2s, msles = [], [], [], []
    for a in [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 2, 5, 10, 20]:
        y_pred, y_test = kf_predict(emb, target, a)
        mae, rmse, r2, msle = compute_metrics(y_pred, y_test)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        msles.append(msle)
    print("MAE: ", np.min(maes))
    print("RMSE: ", np.min(rmses))
    print("R2: ", np.max(r2s))
    print("MSLE: ", np.min(msles))
    return np.min(maes), np.min(rmses), np.max(r2s), np.min(msles)


def run_crime_prediction(emb):
    """
    Predict number of crime incedences
    
    Make use of embeddings matrix to predict crime    
    """
    crime = np.load(data_path+"crime_counts-2015.npy")
    crime = crime[:, 0]

    mae, rmse, r2, msle = predict_crime(emb, crime)

    res = {"mae": mae, "rmse": rmse, "r2": r2, "msle": msle}
    with open('cp.json', 'w') as fp:
        json.dump(res, fp, indent=4)


def get_clusters(emb):
    """
    Fit clusters to the embeddings

    Returns an object with 4 different clusters of the form {cluster_size: clusters}
    """
    res = {}
    for n in range(5, 21, 5):
        kmeans = KMeans(n_clusters=n, random_state=3).fit_predict(emb)
        res[n] = list(kmeans)
    return res

def reformEmbed(ZoneEmbed, Num_zone):
    """
    Reshape the embeddings
    """
    ZoneEmbed_shape = np.shape(ZoneEmbed)

    emb = np.zeros((Num_zone,
                    int(ZoneEmbed_shape[0]/Num_zone*ZoneEmbed_shape[1])))
    for index_1 in range(ZoneEmbed_shape[0]):
        for index_2 in range(ZoneEmbed_shape[1]):
            emb[index_1 % Num_zone, index_2 + ZoneEmbed_shape[1]*int(index_1/Num_zone)] \
                = ZoneEmbed[index_1, index_2]
    return emb


if __name__ == "__main__":
    Num_zone = 180
    ZoneEmbed = np.load('./data/ZoneEmbed.npy')
    emb = reformEmbed(ZoneEmbed, Num_zone)
    print(get_clusters(emb))
    run_crime_prediction(emb)
