"""Main to train and analyze TF models for time-series modeling."""

import os

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data
import flags as cflags  # pylint: disable=unused-import
import plot
import tf_models

FLAGS = flags.FLAGS

flags.DEFINE_enum('cell_type', 'lstm', ['lstm'], 'The type of RNN cell to use.')

flags.DEFINE_integer('cell_units', 16, 'The number of cell units.')

flags.DEFINE_integer('batch_size', 32, 'Batch size.')

flags.DEFINE_integer('input_width', 10,
                     'The width of the input time-series data.')


def make_dataset(data_df, input_width, label_width):
  """Returns a tf.dataset.Dataset for a DataFrame."""
  sequence_length = input_width + label_width
  data = np.array(data_df, dtype=np.float32)
  dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=data, targets=None, sequence_length=sequence_length,
    sequence_stride=1, shuffle=True, batch_size=FLAGS.batch_size)

  @tf.autograph.experimental.do_not_convert
  def split(features):
    inputs = features[:, slice(0, input_width), :]
    labels = features[:, slice(input_width, None), -1]
    inputs.set_shape([None, input_width, None])
    labels.set_shape([None, label_width])
    return inputs, labels

  dataset = dataset.map(split)
  return dataset


def fit_and_forecast(train_dataset, test_dataset, endog_col, exog_cols,
                     forecast_steps):
  """Fits a TF model and returns the fitted values and forecasts."""
  model = tf_models.AutoRegRNN(FLAGS.cell_type, FLAGS.cell_units,
                               forecast_steps)
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='mean_absolute_error', patience=5, mode='min')
  history = model.fit(train_dataset, epochs=20,
                      validation_data=test_dataset,
                      callbacks=[early_stopping])
  return history


def main(argv=()):
  del argv

  filepath = os.path.abspath(FLAGS.filepath)
  start_date, end_date = FLAGS.start_date, FLAGS.end_date
  df = data.read_data_df(FLAGS.dataset_name, filepath,
                         start_date, end_date, FLAGS.data_interval)

  # Train a model.
  endog_col = FLAGS.endog_col_name
  exog_cols = FLAGS.exog_col_names.split(',') if FLAGS.exog_col_names else []
  df = df[exog_cols + [endog_col]]  # endog_col assumed to be the last.
  forecast_steps = FLAGS.forecast_steps
  train_df, test_df = df.iloc[:-forecast_steps], df.iloc[-forecast_steps:]
  print(f'Training dataset size: {len(train_df)}')
  print(f'Test dataset size: {len(test_df)}')

  input_width = FLAGS.input_width
  train_dataset = make_dataset(train_df, input_width, forecast_steps)
  test_dataset = make_dataset(test_df, input_width, forecast_steps)
  history = fit_and_forecast(train_dataset, test_dataset, endog_col,
                             exog_cols, forecast_steps)

  # plot.plot_fit_pred(train_df, test_df, endog_col, fitted_values, predictions)
  # plt.show()


if __name__ == '__main__':
  app.run(main)
