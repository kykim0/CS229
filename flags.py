"""Various top-level flags."""

from absl import flags

# AR models.
flags.DEFINE_integer('lags', None,
                     'The number of lags to include in the AR type models.')

# Exponential smoothing models.
flags.DEFINE_bool('exp_trend', False,
                  'True to use exponential trend. Use linear otherwise.')

flags.DEFINE_bool('damped_trend', True, 'True to damp the trend component.')
