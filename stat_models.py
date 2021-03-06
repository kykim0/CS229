"""Classical statistical time-series models."""

from statsmodels.tsa import api, ar_model, statespace


def simple_exp_smoothing(endog):
  """Fits an simple exponential smoothing model.

  See https://otexts.com/fpp3/ses.html.

  Args:
    endog: (np.ndarray) 1d dependent values.
  """
  model = api.SimpleExpSmoothing(endog=endog)
  # Find the optimal value of alpha automatically.
  model_res = model.fit(optimized=True)
  return model_res


def holt_exp_smoothing(endog, exp_trend=False, damped_trend=False):
  """Fits a Holt's exponential smoothing model.

  See https://otexts.com/fpp3/holt.html.

  Args:
    endog: (np.ndarray) 1d dependent values.
    exp_trend: (bool) True for exponential trend; False for linear trend.
    dapemd_trend: (bool) True to damp trend.
  """
  model = api.Holt(endog=endog, exponential=exp_trend,
                   damped_trend=damped_trend)
  # Optimize the model params automatically.
  model_res = model.fit(optimized=True)
  return model_res


def holt_winters_exp_smoothing(endog, trend='add', damped_trend=False,
                               seasonal='add', seasonal_periods=None,
                               use_boxcox=False):
  """Fits a Holt-Winters model.

  See https://otexts.com/fpp3/holt-winters.html.

  Args:
    endog: (np.ndarray) 1d dependent values.
    trend: (str) type of the trend component: 'add', 'mul'.
    damped_trend: (bool) True to damp trend.
    seasonal: (str) type of the seasonal component: 'add', 'mul'.
    seasonal_periods: (int) number of periods in a complete cycle.
    use_boxcox: (bool) True to apply the Box-Cox transform first.
  """
  model = api.ExponentialSmoothing(endog=endog, trend=trend,
                                   damped_trend=damped_trend, seasonal=seasonal,
                                   seasonal_periods=seasonal_periods,
                                   use_boxcox=use_boxcox)
  # Optimize the model params automatically.
  model_res = model.fit(optimized=True)
  return model_res


def autoreg_model(endog, lags, select_order=False, trend='c', exog=None,
                  period=None):
  """Fits an autoregressive (AR) model.

  Args:
    endog: (np.ndarray) 1d dependent values.
    lags: (int,list[int]) Number of lags to include in the AR model.
    select_order: (bool) if True use `ar_model_order` to select order.
    trend: (str) can be 'n' (none), 'c' (constant), 't' (linear), 'ct' (both)
    exog: (np.ndarray) covariates (exogenous) variables.
    period: (int) period of data, applicable for seasonal data.
  """
  seasonal = (period is not None)
  if select_order:
    ar_select = ar_model.ar_select_order(
      endog, maxlag=lags, ic='aic', glob=False, trend=trend,
      seasonal=seasonal, exog=exog, period=period)
    model = ar_select.model
  else:
    model = ar_model.AutoReg(endog, lags, trend=trend, seasonal=seasonal,
                             exog=exog, period=period)
  model_res = model.fit()
  return model_res


def sarimax_model(endog, exog=None, order=(1, 0, 0), sorder=(0, 0, 0, 0),
                  trend='c'):
  """Fits an ARIMA model with optional seasonality and covariates.

  Args:
    endog: (np.ndarray) 1d dependent values.
    exog: (np.ndarray) optional explanatory variables to model.
    order: (iterable) p, d, q of the non-seasonal component.
    sorder: (iterable) P, D, Q, s of the seasonal component.
    trend: (str) can be 'n' (none), 'c' (constant), 't' (linear), 'ct' (both)
  """
  model = statespace.sarimax.SARIMAX(endog, exog=exog, order=order,
                                     seasonal_order=sorder, trend=trend)
  model_res = model.fit()
  return model_res
