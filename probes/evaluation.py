import torch
import numpy as np
from scipy import stats
from sklearn import metrics


def haversine_distance(true, predicted):
    """
    Compute the haversine distance between the predicted and true latitudes and longitudes.

    :param predicted: (n, 2) array-like, where n is the number of points, and 2 is the dimension (latitude, longitude)
    :param true: (n, 2) array-like, where n is the number of points, and 2 is the dimension (latitude, longitude)
    :return: (n,) array-like, haversine distance for each pair of predicted and true points in kilometers
    """
    R = 6371.0  # Radius of Earth in kilometers

    predicted = np.radians(predicted)
    true = np.radians(true)

    dlat = true[:, 0] - predicted[:, 0]
    dlon = true[:, 1] - predicted[:, 1]

    a = np.sin(dlat/2)**2 + \
        np.cos(predicted[:, 0]) * np.cos(true[:, 0]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def pairwise_haversine_distance(true, predicted):
    """
    Compute the haversine distance between each pair of predicted and true latitudes and longitudes using PyTorch.

    :param predicted: (m, 2) tensor, where m is the number of predicted points, and 2 is the dimension (latitude, longitude)
    :param true: (n, 2) tensor, where n is the number of true points, and 2 is the dimension (latitude, longitude)
    :return: (m, n) tensor, haversine distance for each pair of predicted and true points in kilometers
    """
    R = 6371.0  # Radius of Earth in kilometers

    true = torch.tensor(true, dtype=torch.float32)
    predicted = torch.tensor(predicted, dtype=torch.float32)

    predicted = predicted * (torch.pi / 180)
    true = true * (torch.pi / 180)

    dlat = true[:, None, 0] - predicted[None, :, 0]
    dlon = true[:, None, 1] - predicted[None, :, 1]

    a = torch.sin(dlat / 2) ** 2 + \
        torch.cos(predicted[None, :, 0]) * \
        torch.cos(true[:, None, 0]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return (R * c).numpy()


def haversine_r2(actual_coords, predicted_coords):
    # Step 1: Calculate the mean coordinates
    mean_coords = np.mean(actual_coords, axis=0)

    # Step 2: Calculate SS_tot and SS_res
    ss_tot = np.sum(haversine_distance(actual_coords, mean_coords[None, :])**2)
    ss_res = np.sum(haversine_distance(actual_coords, predicted_coords)**2)

    # Step 3: Calculate R^2
    r2 = 1 - (ss_res / ss_tot)

    return r2


def pairwise_abs_distance_fn(true_values, predicted_values):
    return np.abs(true_values[:, np.newaxis] - predicted_values[:, np.newaxis].T)


def compute_proximity_error_matrix(true_values, predicted_values, distance_fn):
    if len(true_values) != len(predicted_values):
        raise ValueError("Input arrays must have the same length")

    dist_matrix = distance_fn(true_values, predicted_values)
    target_diff = np.diag(dist_matrix)
    error_matrix = dist_matrix < target_diff[:, np.newaxis]
    return error_matrix


def proximity_scores(error_matrix, is_test):
    train_error = error_matrix[~is_test, :][:, ~is_test].mean(axis=1)
    test_error = error_matrix[is_test, :][:, is_test].mean(axis=1)
    combined_error = error_matrix.mean(axis=1)
    return train_error, test_error, combined_error


def score_place_probe(target, pred, use_haversine=False):
    x_pearson = stats.pearsonr(target[:, 0], pred[:, 0])
    x_spearman = stats.spearmanr(target[:, 0], pred[:, 0])
    x_kendall = stats.kendalltau(target[:, 0], pred[:, 0])

    y_pearson = stats.pearsonr(target[:, 1], pred[:, 1])
    y_spearman = stats.spearmanr(target[:, 1], pred[:, 1])
    y_kendall = stats.kendalltau(target[:, 1], pred[:, 1])

    score_dict = {
        'x_r2': metrics.r2_score(target[:, 0], pred[:, 0]),
        'y_r2': metrics.r2_score(target[:, 1], pred[:, 1]),
        'r2': metrics.r2_score(target, pred),
        'x_mae': metrics.mean_absolute_error(target[:, 0], pred[:, 0]),
        'y_mae': metrics.mean_absolute_error(target[:, 1], pred[:, 1]),
        'mae': metrics.mean_absolute_error(target, pred),
        'mse': metrics.mean_squared_error(target, pred),
        'rmse': np.sqrt(metrics.mean_squared_error(target, pred)),
        'x_pearson': x_pearson.correlation,
        'x_pearson_p': x_pearson.pvalue,
        'x_spearman': x_spearman.correlation,
        'x_spearman_p': x_spearman.pvalue,
        'x_kendall': x_kendall.correlation,
        'x_kendall_p': x_kendall.pvalue,
        'y_pearson': y_pearson.correlation,
        'y_pearson_p': y_pearson.pvalue,
        'y_spearman': y_spearman.correlation,
        'y_spearman_p': y_spearman.pvalue,
        'y_kendall': y_kendall.correlation,
        'y_kendall_p': y_kendall.pvalue
    }
    if use_haversine:
        hav_dist = haversine_distance(target, pred)
        score_dict['haversine_mse'] = np.mean(hav_dist**2)
        score_dict['haversine_rmse'] = np.sqrt(np.mean(hav_dist**2))
        score_dict['haversine_mae'] = np.mean(hav_dist)
        score_dict['haversine_r2'] = haversine_r2(target, pred)
    return score_dict


def score_time_probe(target, pred):
    pearson = stats.pearsonr(target, pred)
    spearman = stats.spearmanr(target, pred)
    kendall = stats.kendalltau(target, pred)
    score_dict = {
        'mae': metrics.mean_absolute_error(target, pred),
        'mse': metrics.mean_squared_error(target, pred),
        'rmse': np.sqrt(metrics.mean_squared_error(target, pred)),
        'r2': metrics.r2_score(target, pred),
        'pearson': pearson.correlation,
        'pearson_p': pearson.pvalue,
        'spearman': spearman.correlation,
        'spearman_p': spearman.pvalue,
        'kendall': kendall.correlation,
        'kendall_p': kendall.pvalue
    }
    return score_dict
