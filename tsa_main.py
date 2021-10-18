"""Main to train and analyze time-series models.

TODO(kykim):
- Allow to sample data (now only 10 mins)
"""

import os

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import seaborn as sns

import data
import flags as cflags
import metrics
import stat_models

sns.set_style('darkgrid')

FLAGS = flags.FLAGS

flags.DEFINE_enum('dataset_name', 'weather', ['weather'],
                  'Name of the dataset for time-series modeling.')

flags.DEFINE_string('filepath', 'data/jena_climate_2009_2016.csv',
                    'Path to a csv data file.')

flags.DEFINE_enum('model', None,
                  ['ar', 'ma', 'arma', 'arima', 'sarima',
                   'ses', 'holt', 'holtwinters'],
                  'The type of time-series model to use.')

flags.DEFINE_string('start_date', None, 'Start date for the data.')

flags.DEFINE_string('end_date', None, 'End date for the data.')

flags.DEFINE_integer('forecast_steps', 10,
                     'The number of steps to forecast. This many data points'
                     ' from the end are used for testing.')

flags.DEFINE_string('dependent_col_name', 'T (degC)',
                    'The name of the dependent variable column.')

flags.DEFINE_bool('plot_diagnostics', False,
                  'True to plot model diagnostics.')


def fit_and_forecast(model, train_df, test_df, tcolumn, forecast_steps):
  """Fits a stats model and returns fitted values and forecasts."""
  y_true = train_df[tcolumn].to_numpy()
  train_len, test_len = len(train_df.index), len(test_df.index)

  model_res, fitted_values, predictions = None, None, None
  if model == 'ar':
    lags = FLAGS.lags
    model_res = stat_models.autoreg_model(y_true, lags=lags, select_order=True)
  elif model == 'ses':
    model_res = stat_models.simple_exp_smoothing(y_true)
  elif model == 'holt':
    exp_trend = FLAGS.exp_trend
    damped_trend = FLAGS.damped_trend
    model_res = stat_models.holt_exp_smoothing(y_true, exp_trend=exp_trend,
                                               damped_trend=damped_trend)
  elif model == 'holtwinters':
    model_res = stat_models.holt_winters_exp_smoothing()

  fitted_values = model_res.fittedvalues
  predictions = model_res.forecast(forecast_steps)
  print(y_true[:30])
  print(fitted_values[:30])
  print(model_res.params)
  print(model_res.summary())
  return model_res, fitted_values, predictions
  

def plot_fit_pred(train_df, test_df, tcolumn, fitted_values, predictions):
  """Creates a plot of fitted values and predictions

  Args:
    train_df: (DataFrame) training data.
    test_df: (DataFrame) test data.
    tcolumn: (str) name of the depedent variable column.
    fitted_values: (np.ndarray) in-sample fitted values.
    predictions: (np.ndarray) out-of-sample predictions.
  """
  # Plot the training and test data.
  fig = plt.figure(figsize=(10, 6))
  ax = fig.subplots()
  ax.set_title(tcolumn)
  ax.plot(train_df.index, train_df[tcolumn])
  ax.plot(test_df.index, test_df[tcolumn])

  # Plot the fitted values and predictions.
  # Note that the length of `fitted_values` may not match the size of the
  # training dataset if, for instance, a long lag was used for an AR model.
  ax.plot(train_df.index[-len(fitted_values):], fitted_values, '--')
  ax.plot(test_df.index, predictions, '--')
  ax.legend(labels=['Train', 'Test', 'Fit', 'Pred'])
  return fig


def main(argv=()):
  del argv

  filepath = os.path.abspath(FLAGS.filepath)
  start_date, end_date = FLAGS.start_date, FLAGS.end_date
  df = data.read_data_df(FLAGS.dataset_name, filepath,
                         start_date, end_date)

  # Train a model.
  tcolumn = FLAGS.dependent_col_name
  forecast_steps = FLAGS.forecast_steps
  train_df, test_df = df.iloc[:-forecast_steps], df.iloc[-forecast_steps:]
  model_res, fitted_values, predictions = fit_and_forecast(
    FLAGS.model, train_df, test_df, tcolumn, forecast_steps)

  print(test_df[tcolumn].to_numpy()[-30:])
  print(predictions[-30:])
  mae = metrics.mean_absolute_error(test_df[tcolumn].to_numpy(), predictions)
  print(f'Mean absolute error: {mae}')

  plot_fit_pred(train_df, test_df, tcolumn, fitted_values, predictions)
  if FLAGS.plot_diagnostics:
    fig = plt.figure(figsize=(16, 9))
    model_res.plot_diagnostics(fig=fig)
  plt.show()


if __name__ == '__main__':
  app.run(main)
