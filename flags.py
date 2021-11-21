"""Common flags."""

from absl import flags

# Logistics.
flags.DEFINE_integer('forecast_steps', 10,
                     'The number of steps to forecast. This many data points'
                     ' from the end are used for testing.')

flags.DEFINE_string('endog_col_name', 'T (degC)',
                    'The name of the dependent variable column.')

flags.DEFINE_string('exog_col_names', '',
                    'The names of the explanatory variable columns if any.')

flags.DEFINE_string('filepath', 'data/jena_climate_2009_2016.csv',
                    'Path to a csv data file.')

flags.DEFINE_enum('dataset_name', 'weather', ['weather'],
                  'Name of the dataset for time-series modeling.')

flags.DEFINE_string('start_date', None, 'Start date for the data.')

flags.DEFINE_string('end_date', None, 'End date for the data.')

flags.DEFINE_integer('data_interval', None,
                     'If not None, only include data points at this interval.')
