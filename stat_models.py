"""Classical statistical time-series models."""

from statsmodels.tsa.ar_model import AutoReg, ar_select_order


def autoreg_model(endog, lags, select_order=False, trend='c', exog=None,
                  period=None):
  """Returns an autoregressive (AR) model.

  Args:
    endog: (np.array) 1d dependent values.
    lags: (int, list[int]) Number of lags to include in the AR model.
    select_order: (bool) if True use `ar_model_order` to select order.
    exog: (np.array) covariates (exogenous) variables.
    period: (int) period of data, applicable for seasonal data.
  """
  seasonal = (period is not None)
  if select_order:
    ar_select = ar_select_order(endog, maxlag=lags, ic='aic', glob=False,
                                trend=trend, seasonal=seasonal, exog=exog,
                                period=period)
    model = ar_select.model
  else:
    model = AutoReg(endog, lags, trend=trend, seasonal=seasonal, exog=exog,
                    period=period)
  model_res = model.fit()
  return model_res
