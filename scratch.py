import os

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import data
import flags as cflags  # pylint: disable=unused-import

sns.set(font_scale=0.7)
sns.set_style('darkgrid')

FLAGS = flags.FLAGS


def main(argv=()):
  del argv

  filepath = os.path.abspath(FLAGS.filepath)
  start_date, end_date = FLAGS.start_date, FLAGS.end_date
  df = data.read_data_df(FLAGS.dataset_name, filepath,
                         start_date, end_date)

  corr = df.corr(method='spearman')
  sns.heatmap(corr.round(2), annot=True, linewidths=.5)
  plt.show()


if __name__ == '__main__':
  app.run(main)
