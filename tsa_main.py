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

import data
import flags as cflags  # pylint: disable=unused-import
import metrics
import plot
import stat_models

FLAGS = flags.FLAGS

flags.DEFINE_enum('model', None,
                  ['ar', 'sarimax', 'ses', 'holt', 'holtwinters'],
                  'The type of time-series model to use.')

flags.DEFINE_bool('plot_diagnostics', False,
                  'True to plot model diagnostics.')

# AR models.
flags.DEFINE_integer('lags', None,
                     'The number of lags to include in the AR type models.')

flags.DEFINE_enum('arima_trend', 'c', ['n', 'c', 't', 'ct'],
                  'Type of trend to model for ARIMA models.')

flags.DEFINE_string('arima_p', '1',
                    'The p parameter for the non-seasonal component. This can'
                    ' be a csv for specific orders.')

flags.DEFINE_integer('arima_d', 0,
                     'The d parameter for the non-seasonal component.')

flags.DEFINE_string('arima_q', '0',
                    'The q parameter for the non-seasonal component. This can'
                    ' be a csv for specific orders.')

flags.DEFINE_string('sarima_p', '0',
                    'The p parameter for the non-seasonal component. This can'
                    ' be a csv for specific orders.')

flags.DEFINE_integer('sarima_d', 0,
                     'The d parameter for the non-seasonal component.')

flags.DEFINE_string('sarima_q', '0',
                    'The q parameter for the non-seasonal component. This can'
                    ' be a csv for specific orders.')

flags.DEFINE_integer('sarima_s', 0, 'The seasonal periodicity.')

# Exponential smoothing models.
flags.DEFINE_bool('exp_trend', False,
                  'True to use exponential trend. Use linear otherwise.')

flags.DEFINE_bool('damped_trend', True, 'True to damp the trend component.')

flags.DEFINE_enum('ets_trend', 'add', ['add', 'mul'],
                  'Type of the trend component to use.')

flags.DEFINE_enum('seasonal', 'add', ['add', 'mul'],
                  'Type of the seasonal component to use.')

flags.DEFINE_integer('seasonal_periods', None,
                     'Number of periods in a complete seasonal cycle.')

flags.DEFINE_bool('use_boxcox', False, 'True to apply the Box-Cox transform.')


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
  if exog_test is not None:
    predictions = model_res.forecast(steps=forecast_steps, exog=exog_test)
  else:
    predictions = model_res.forecast(steps=forecast_steps)
  return model_res, fitted_values, predictions


def main(argv=()):
  del argv

  filepath = os.path.abspath(FLAGS.filepath)
  start_date, end_date = FLAGS.start_date, FLAGS.end_date
  df = data.read_data_df(FLAGS.dataset_name, filepath,
                         start_date, end_date, FLAGS.data_interval)

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

  plot.plot_fit_pred(train_df, test_df, endog_col, fitted_values, predictions)
  if FLAGS.plot_diagnostics:
    fig = plt.figure(figsize=(16, 9))
    model_res.plot_diagnostics(fig=fig)
  plt.show()


if __name__ == '__main__':
  app.run(main)
