"""Main to train and analyze time-series models.

Example usage:
  $ python3 tsa_main.py \
      --model=ses --start_date=2010-01-01 --end_date=2012-01-31 \
      --forecast_steps=2 --data_interval=4320
"""

import os

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import seaborn as sns

import data
import flags as cflags  # pylint: disable=unused-import
import metrics
import stat_models

sns.set_style('darkgrid')

FLAGS = flags.FLAGS

flags.DEFINE_enum('model', None,
                  ['ar', 'sarimax', 'ses', 'holt', 'holtwinters'],
                  'The type of time-series model to use.')

flags.DEFINE_integer('forecast_steps', 10,
                     'The number of steps to forecast. This many data points'
                     ' from the end are used for testing.')

flags.DEFINE_string('endog_col_name', 'T (degC)',
                    'The name of the dependent variable column.')

flags.DEFINE_string('exog_col_names', '',
                    'The names of the explanatory variable columns if any.')

flags.DEFINE_bool('plot_diagnostics', False,
                  'True to plot model diagnostics.')


def _parse(csv_str):
  """Util to parse a comma-separated value in string."""
  str_l = list(map(int, csv_str.split(',')))
  return str_l[0] if len(str_l) == 1 else tuple(str_l)


def fit_and_forecast(model, train_df, test_df, endog_col, exog_cols,
                     forecast_steps):
  """Fits a stats model and returns fitted values and forecasts."""
  y_true = train_df[endog_col].to_numpy()

  model_res, fitted_values, predictions = None, None, None
  if model == 'ar':
    lags = FLAGS.lags
    model_res = stat_models.autoreg_model(y_true, lags=lags, select_order=True)
  elif model == 'sarimax':
    exog = train_df[exog_cols].to_numpy() if exog_cols else None
    order = tuple([_parse(FLAGS.arima_p), FLAGS.arima_d, _parse(FLAGS.arima_q)])
    sorder = tuple([_parse(FLAGS.sarima_p), FLAGS.sarima_d,
                    _parse(FLAGS.sarima_q), FLAGS.sarima_s])
    trend = FLAGS.arima_trend
    model_res = stat_models.sarimax_model(y_true, exog=exog, order=order,
                                          sorder=sorder, trend=trend)
  elif model == 'ses':
    model_res = stat_models.simple_exp_smoothing(y_true)
  elif model == 'holt':
    exp_trend = FLAGS.exp_trend
    damped_trend = FLAGS.damped_trend
    model_res = stat_models.holt_exp_smoothing(y_true, exp_trend=exp_trend,
                                               damped_trend=damped_trend)
  elif model == 'holtwinters':
    trend = FLAGS.ets_trend
    damped_trend = FLAGS.damped_trend
    seasonal = FLAGS.seasonal
    periods = FLAGS.seasonal_periods
    use_boxcox = FLAGS.use_boxcox
    model_res = stat_models.holt_winters_exp_smoothing(
      y_true, trend=trend, damped_trend=damped_trend, seasonal=seasonal,
      seasonal_periods=periods, use_boxcox=use_boxcox)
  else:
    raise ValueError(f'Unsupported model type {model}')

  print(model_res.params)
  print(model_res.summary())

  fitted_values = model_res.fittedvalues
  exog_test = test_df[exog_cols].to_numpy() if exog_cols else None
  predictions = model_res.forecast(steps=forecast_steps, exog=exog_test)
  return model_res, fitted_values, predictions


def plot_fit_pred(train_df, test_df, endog_col, fitted_values, predictions):
  """Creates a plot of fitted values and predictions

  Args:
    train_df: (DataFrame) training data.
    test_df: (DataFrame) test data.
    endog_col: (str) name of the depedent variable column.
    fitted_values: (np.ndarray) in-sample fitted values.
    predictions: (np.ndarray) out-of-sample predictions.
  """
  # Plot the training and test data.
  fig = plt.figure(figsize=(10, 6))
  ax = fig.subplots()
  ax.set_title(endog_col)
  ax.plot(train_df.index, train_df[endog_col], '.-')
  ax.plot(test_df.index, test_df[endog_col], '.-')

  # Plot the fitted values and predictions.
  # Note that the length of `fitted_values` may not match the size of the
  # training dataset if, for instance, a long lag was used for an AR model.
  ax.plot(train_df.index[-len(fitted_values):], fitted_values, '.--')
  ax.plot(test_df.index, predictions, '.--')
  ax.legend(labels=['Train', 'Test', 'Fit', 'Pred'])
  return fig


def main(argv=()):
  del argv

  filepath = os.path.abspath(FLAGS.filepath)
  start_date, end_date = FLAGS.start_date, FLAGS.end_date
  df = data.read_data_df(FLAGS.dataset_name, filepath,
                         start_date, end_date)
  if FLAGS.data_interval:
    df = df[::FLAGS.data_interval]

  # Train a model.
  endog_col = FLAGS.endog_col_name
  exog_cols = FLAGS.exog_col_names.split(',') if FLAGS.exog_col_names else None
  forecast_steps = FLAGS.forecast_steps
  train_df, test_df = df.iloc[:-forecast_steps], df.iloc[-forecast_steps:]
  print(f'Training dataset size: {len(train_df)}')
  print(f'Test dataset size: {len(test_df)}')
  model_res, fitted_values, predictions = fit_and_forecast(
    FLAGS.model, train_df, test_df, endog_col, exog_cols, forecast_steps)

  mae = metrics.mean_absolute_error(test_df[endog_col].to_numpy(), predictions)
  print(f'Mean absolute error: {mae}')

  plot_fit_pred(train_df, test_df, endog_col, fitted_values, predictions)
  if FLAGS.plot_diagnostics:
    fig = plt.figure(figsize=(16, 9))
    model_res.plot_diagnostics(fig=fig)
  plt.show()


if __name__ == '__main__':
  app.run(main)
