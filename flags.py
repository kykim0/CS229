"""Various top-level flags."""

from absl import flags

# Data.
flags.DEFINE_string('filepath', 'data/jena_climate_2009_2016.csv',
                    'Path to a csv data file.')

flags.DEFINE_enum('dataset_name', 'weather', ['weather'],
                  'Name of the dataset for time-series modeling.')

flags.DEFINE_string('start_date', None, 'Start date for the data.')

flags.DEFINE_string('end_date', None, 'End date for the data.')

flags.DEFINE_integer('data_interval', None,
                     'If not None, only include data points at this interval.')

# AR models.
flags.DEFINE_integer('lags', None,
                     'The number of lags to include in the AR type models.')

# Exponential smoothing models.
flags.DEFINE_bool('exp_trend', False,
                  'True to use exponential trend. Use linear otherwise.')

flags.DEFINE_bool('damped_trend', True, 'True to damp the trend component.')

flags.DEFINE_enum('trend', 'add', ['add', 'mul'],
                  'Type of the trend component to use.')

flags.DEFINE_enum('seasonal', 'add', ['add', 'mul'],
                  'Type of the seasonal component to use.')

flags.DEFINE_integer('seasonal_periods', None,
                     'Number of periods in a complete seasonal cycle.')

flags.DEFINE_bool('use_boxcox', False, 'True to apply the Box-Cox transform.')
