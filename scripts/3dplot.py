#!/usr/bin/env python3

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np


def main(args):
  with open(args.stats, 'r') as fd:
    data = json.load(fd)

  # Turns the keys back into integers
  ndata = {}
  for tp in data.keys():
    tpi = int(tp)
    ndata[tpi] = {}
    for pp in data[tp].keys():
      ppi = int(pp)
      ndata[tpi][ppi] = data[tp][pp]
  data = ndata

  print(data)

  tps = sorted(list(data.keys()))
  pps = set()
  for tp in data.keys():
    for pp in data[tp].keys():
      pps.add(pp)
  pps = sorted(list(pps))
  print(tps)
  print(pps)
  fdata = np.full((len(pps), len(tps)), float('NaN'))
  for tp in data.keys():
    for pp in data[tp].keys():
      if data[tp][pp]:
        fdata[pps.index(pp)][tps.index(tp)] = data[tp][pp]
  print(fdata)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X, Y = np.meshgrid(list(range(len(tps))), list(range(len(pps))))
  print(fdata)
  ax.plot_surface(X, Y, fdata, rstride=1, cstride=1,
                  cmap='rainbow',
                  edgecolor='none')
  ax.set_xlabel('Tensor Parallelism')
  ax.set_ylabel('Pipeline Parallelism')
  ax.set_zlabel('Batch Time (s)')
  ax.set_title(f'Parallelism Search')
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
  args = ap.parse_args()
  main(args)
