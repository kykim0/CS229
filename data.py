"""Data processing for time-series modeling."""

import datetime

import numpy as np
import pandas as pd


def _read_weather_df(filepath, start=None, end=None):
  """Returns a DataFrame of the processed weather data."""
  df = pd.read_csv(filepath)

  # Convert to datetime and filter rows based on time if applicable.
  df['Date Time'] = pd.to_datetime(
    df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
  start_datetime = df['Date Time'][0]
  if start is not None:
    start_datetime = datetime.datetime.strptime(start, '%Y-%m-%d')
    df = df[df['Date Time'] >= start_datetime]
  end_datetime = df['Date Time'][df.index[-1]]
  if end is not None:
    end_datetime = datetime.datetime.strptime(end, '%Y-%m-%d')
    df = df[df['Date Time'] <= end_datetime]

  # Fix wind velocities of -9999.0.
  df.loc[df['wv (m/s)'] == -9999.0, 'wv (m/s)'] = 0.0
  df.loc[df['max. wv (m/s)'] == -9999.0, 'max. wv (m/s)'] = 0.0

  # Decompose wind velocities into x, y components.
  wv, max_wv = df.pop('wv (m/s)'), df.pop('max. wv (m/s)')
  wd_rad = df.pop('wd (deg)') * np.pi / 180.0
  df['wvx'] = wv * np.cos(wd_rad)
  df['wvy'] = wv * np.sin(wd_rad)
  df['max wvx'] = max_wv * np.cos(wd_rad)
  df['max wvy'] = max_wv * np.sin(wd_rad)

  # Add time frequency features.
  timestamp = df['Date Time'].map(pd.Timestamp.timestamp)
  df['dsin'] = np.sin(timestamp * (2 * np.pi / 86400))
  df['dcos'] = np.cos(timestamp * (2 * np.pi / 86400))
  df['ysin'] = np.sin(timestamp * (2 * np.pi / (365 * 86400)))
  df['ycos'] = np.cos(timestamp * (2 * np.pi / (365 * 86400)))

  df = df.set_index('Date Time')
  return df


def read_data_df(name, filepath, start=None, end=None):
  """Returns time-series data as a DataFrame instance.

  Args:
    name: (str) name of the dataset. Supported dataset names include
      `weather`.
    filepath: (str) absolute path to the dataset in csv.
    start: (str) start of the time-series.
    end: (str) end of the time-series.
  """
  if name == 'weather':
    return _read_weather_df(filepath, start, end)
  raise ValueError(f'Unsupported dataset {name}')
