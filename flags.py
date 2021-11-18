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
