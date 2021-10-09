"""Metrics to compare time-series models."""

import numpy as np
import sklearn


def mean_sqaured_error(y_true, y_pred):
  """Returns the average of squared errors."""
  return sklearn.metrics.mean_sqaured_error(y_true, y_pred)


def mean_squared_log_error(y_true, y_pred):
  """Returns the average of log of squared errors."""
  return sklearn.metrics.mean_squared_log_error(y_true, y_pred)


def mean_absolute_error(y_true, y_pred):
  """Returns the average of the absolute values of errors."""
  return sklearn.metrics.mean_absolute_error(y_true, y_pred)


def mean_absolute_percent_error(y_true, y_pred):
  """Returns mean absolute error in percentage."""
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
