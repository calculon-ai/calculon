#!/usr/bin/env python3

import argparse
import calculon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import tol_colors as tc


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
  fdata = np.full((len(tps), len(pps)), float('NaN'))
  for tp in data.keys():
    for pp in data[tp].keys():
      if 'stats' in data[tp][pp]:
        v = data[tp][pp]['stats']['sample_rate']
        fdata[tps.index(tp)][pps.index(pp)] = v
        print(f'{tp},{pp} is {v}')
      else:
        print(f'{tp},{pp} has none')

  # Determines range
  minf = min(map(min, fdata))
  maxf = max(map(max, fdata))
  black_threshold = minf + (maxf - minf) * 0.30
  print(f'min={minf} max={maxf} thres={black_threshold}')

  # Creates the plot
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(fdata, origin='lower', cmap='hot')#, linewidth=0.5)
  ax.set_xticks(np.arange(len(pps)), labels=pps)
  ax.set_xlabel('Pipeline Parallelism')
  ax.set_yticks(np.arange(len(tps)), labels=tps)
  ax.set_ylabel('Tensor Parallelism')
  for tp in tps:
    for pp in pps:
      perf = fdata[tps.index(tp), pps.index(pp)]
      color = 'black' if perf > black_threshold else 'white'
      perf = f'{perf:.1f}'
      text = f'{perf}'
      ax.text(pps.index(pp), tps.index(tp), text, ha='center', va='center',
              color=color)
  if args.title:
    ax.set_title(args.title)
  print(f'writing {args.output}')
  fig.tight_layout()
  fig.savefig(args.output)
  plt.close(fig)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('stats', type=str,
                  help='File path to stats input')
  ap.add_argument('output', type=str,
                  help='Output plot file')
  ap.add_argument('-t', '--title', type=str, default=None,
                  help='Title of plot')
  args = ap.parse_args()
  main(args)
