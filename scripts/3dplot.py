#!/usr/bin/env python3

import argparse
import calculon
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np


def main(args):
  data = calculon.io.read_json_file(args.stats)

  # Turns the keys back into integers
  ndata = {}
  for tp in data.keys():
    tpi = int(tp)
    ndata[tpi] = {}
    for pp in data[tp].keys():
      ppi = int(pp)
      ndata[tpi][ppi] = data[tp][pp]
  data = ndata
  tps = sorted(list(data.keys()))
  pps = set()
  for tp in data.keys():
    for pp in data[tp].keys():
      pps.add(pp)
  pps = sorted(list(pps))
  assert len(tps) > 1, f'len(tps)={len(tps)} can\'t plot'
  assert len(pps) > 1, f'len(pps)={len(pps)} can\'t plot'

  # Gathers data
  fdata = np.full((len(pps), len(tps)), float('NaN'))
  for tp in data.keys():
    for pp in data[tp].keys():
      if 'stats' in data[tp][pp]:
        v = data[tp][pp]['stats']['sample_rate']
        fdata[pps.index(pp)][tps.index(tp)] = v
        print(f'{tp},{pp} is {v}')
      else:
        print(f'{tp},{pp} has none')

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X, Y = np.meshgrid(list(range(len(tps))), list(range(len(pps))))
  ax.plot_surface(X, Y, fdata, rstride=1, cstride=1,
                  cmap='rainbow',
                  edgecolor='none')
  ax.set_xlabel('Tensor Parallelism')
  ax.set_ylabel('Pipeline Parallelism')
  ax.set_zlabel('Sample Rate (s/sec)')
  if args.title:
    ax.set_title(args.title)
  ax.view_init(20, 180+25)
  @tkr.FuncFormatter
  def formatter(x, pos):
    d = 2**x
    if d < 1:
      return 'duh'
    else:
      return str(int(d))
  ax.xaxis.set_major_formatter(formatter)
  ax.yaxis.set_major_formatter(formatter)
  fig.tight_layout()
  plt.show()



if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('stats', type=str,
                  help='File path to stats input')
  ap.add_argument('-t', '--title', type=str, default=None,
                  help='Title of plot')
  args = ap.parse_args()
  main(args)
