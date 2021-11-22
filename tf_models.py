"""TensorFlow models for modeling time-series."""

import tensorflow as tf


class AutoRegRNN(tf.keras.Model):
  """Autoregresive RNN.

  Runs an RNN cell in an autoregressive manner.
  """

  def __init__(self, cell_type, units, out_dim, forecast_steps):
    super().__init__()
    self.units = units
    self.forecast_steps = forecast_steps
    if cell_type == 'lstm':
      self.rnn_cell = tf.keras.layers.LSTMCell(units)
    else:
      raise ValueError(f'Unsupported RNN cell type {cell_type}')
    self.rnn = tf.keras.layers.RNN(self.rnn_cell, return_state=True)
    # Project the RNN output to predictions of all variables.
    self.dense = tf.keras.layers.Dense(out_dim)


  def init(self, inputs):
    """Initializes the RNN state.

    Args:
      inputs: (Tensor) input tensor of shape [batch, time, features].

    Returns:
      A tuple of of a prediction and RNN state.
    """
    x, *states = self.rnn(inputs)
    prediction = self.dense(x)  # [batch, out_dim]
    return prediction, states


  def call(self, inputs, training=None):
    """Runs the RNN in an autoregressive manner to compute predictions.

    Args:
      inputs: (Tensor) input tensor of shape [batch, time, features].
      training: (bool) true to run in training mode.

    Returns:
      A tensor containing predictions of size [batch, time].
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

    predictions = tf.stack(predictions)  # [time, batch, out_dim]
    predictions = tf.transpose(predictions, [1, 0, 2])  # [batch, time, out_dim]
    return predictions
