#!/usr/bin/env python3

import argparse
import json
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('directory', type=str, help='Directory of sweep')
ap.add_argument('batch_mode', type=int, help='0 for num_procs, >0 for max')
args = ap.parse_args()

assert os.path.isdir(args.directory)
os.makedirs(os.path.join(args.directory, 'results'), exist_ok=True)

configs = os.path.join(args.directory, 'configs.json')
assert os.path.isfile(configs)
with open(configs, 'r') as fd:
  configs = json.load(fd)

print(f'name,batch_size,batch_time,sample_rate,ceff,seff,teff,mem1,mem2,off_bw,tp,pp,dp,tn,pn,dn,pi,mbs,recompute,tp_comm,redo')
for name, nodes in configs:
  config = os.path.join(args.directory, f'{name}.json')
  exe = os.path.join(args.directory, 'results', f'{name}_exe.json')
  stats = os.path.join(args.directory, 'results', f'{name}_stats.json')
  raw = os.path.join(args.directory, 'results', f'{name}_raw.json')

  if args.batch_mode <= 0:
    max_batch_size = nodes
  else:
    max_batch_size = args.batch_mode

  if not os.path.isfile(stats):
    print(f'Running {name}')
    cmd = (f'PYTHONPATH=. ./bin/calculon megatron-optimal-execution '
           f'examples/1T.json {nodes} {max_batch_size} {config} '
           f'-e {exe} -s {stats} -r {raw} -c 32')
    subprocess.run(cmd, shell=True, check=True)
    print('done\n')

  with open(exe, 'r') as fd:
    e = json.load(fd)
    batch_size = e['batch_size']
    tp = e['tensor_par']
    pp = e['pipeline_par']
    dp = e['data_par']
    tn = e['tensor_par_net']
    pn = e['pipeline_par_net']
    dn = e['data_par_net']
    pi = e['pipeline_interleaving']
    mbs = e['microbatch_size']
    ar = e['activation_recompute']
    tp_comm = e['tensor_par_comm_type']
    redo = e['seq_par_ag_redo']
    with open(stats, 'r') as fd:
      s = json.load(fd)
      batch_time = s['total_time']
      sample_rate = batch_size / batch_time
      ceff = s['compute_efficiency']
      seff = s['system_efficiency']
      teff = s['total_efficiency']
      mem1 = s['proc_mem_tier1_cap_req'] / 1024**3
      mem2 = s['proc_mem_tier2_cap_req'] / 1024**3
      off_bw = s['offload_mem_bw_req'] / 1e9
      print(f'{name},{batch_size},{batch_time},{sample_rate},{ceff},{seff},{teff},{mem1},{mem2},{off_bw},{tp},{pp},{dp},{tn},{pn},{dn},{pi},{mbs},{ar},{tp_comm},{redo}')
