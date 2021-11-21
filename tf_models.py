"""TensorFlow models for modeling time-series."""

import tensorflow as tf


class AutoRegRNN(tf.keras.Model):
  """Autoregresive RNN.

  Runs an RNN cell in an autoregressive manner.
  """

  def __init__(self, cell_type, units, forecast_steps):
    super().__init__()
    self.units = units
    self.forecast_steps = forecast_steps
    if cell_type == 'lstm':
      self.rnn_cell = tf.keras.layers.LSTMCell(units)
    else:
      raise ValueError(f'Unsupported RNN cell type {cell_type}')
    self.rnn = tf.keras.layers.RNN(self.rnn_cell, return_state=True)
    # Project the RNN output to a prediction for a single dependent variable.
    self.dense = tf.keras.layers.Dense(1)


  def init(self, inputs):
    """Initializes the RNN state.

    Args:
      inputs: (Tensor) input tensor of shape [batch_size, time, features].

    Returns:
      A tuple of of a prediction and RNN state.
    """
    x, *states = self.rnn(inputs)
    prediction = self.dense(x)  # [batch_size, 1]
    return prediction, states


  def call(self, inputs, training=None):
    """Runs the RNN in an autoregressive manner to compute predictions.

    Args:
      inputs: (Tensor) input tensor of shape [batch_size, time, features].
      training: (bool) true to run in training mode.

    Returns:
      A tensor containing predictions of size [batch_size, time].
    """
    prediction, states = self.init(inputs)
    predictions = [prediction]

    # Run the RNN for the prediction steps.
    for _ in range(1, self.forecast_steps):
      # Use the last prediction as input.
      x = prediction
      # Execute one RNN step.
      x, states = self.rnn_cell(x, states=states, training=training)
      # Project the output to a prediction.
      prediction = self.dense(x)
      predictions.append(prediction)

    predictions = tf.stack(predictions)  # [time, batch_size, 1]
    predictions = tf.squeeze(tf.transpose(predictions, [1, 0, 2]))
    return predictions
