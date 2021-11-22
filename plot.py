"""Utils for plotting."""

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')


def plot_fit_pred(train_df, test_df, endog_col,
                  fitted_values=None, predictions=None):
  """Creates a plot of fitted values and predictions.

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
  labels = ['Train', 'Test']

  # Plot the fitted values and predictions.
  # Note that the length of `fitted_values` may not match the size of the
  # training dataset if, for instance, a long lag was used for an AR model.
  if fitted_values is not None:
    ax.plot(train_df.index[-len(fitted_values):], fitted_values, '.--')
    labels.append('Fit')
  if predictions is not None:
    ax.plot(test_df.index, predictions, '.--')
    labels.append('Pred')
  ax.legend(labels=labels)
  return fig
