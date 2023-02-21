#!/usr/bin/env python3

import argparse
import json
import os
import subprocess


def get_executions(logfile, verbose):
  if verbose:
    print(f'  Opening {logfile}')
  with open(logfile, 'r') as fd:
    lines = fd.readlines()
  total = int(lines[0].strip().split()[-1])
  good = int(lines[1].strip().split()[-1])
  bad = int(lines[2].strip().split()[-1])
  return total, good, bad


def main(args):
  if not os.path.isdir(args.directory):
    os.makedirs(args.directory, exist_ok=True)

  assert os.path.isfile(args.configs)
  with open(args.configs, 'r') as fd:
    configs = json.load(fd)

  csv_file = os.path.join(args.directory, 'syscomp.csv')
  for config, nodes in configs:
    assert os.path.isfile(config), f'"{config}" does not exist'
    name = os.path.splitext(os.path.basename(config))[0]
    log = os.path.join(args.directory, f'{name}.log')
    exe = os.path.join(args.directory, f'{name}_exe.json')
    stats = os.path.join(args.directory, f'{name}_stats.json')
    raw = os.path.join(args.directory, f'{name}_raw.json')

    if args.batch_mode <= 0:
      max_batch_size = nodes
    else:
      max_batch_size = args.batch_mode

    if args.csvonly and not os.path.isfile(stats):
      print(f'{stats} does not exist')
      sys.exit(-1)
    elif not os.path.isfile(stats):
      print(f'Running {name}')
      cmd = (f'PYTHONPATH=. ./bin/calculon megatron-optimal-execution '
             f'{args.application} {nodes} {max_batch_size} {config} '
             f'-e {exe} -s {stats} -r {raw}')
      with open(log, 'w') as log_fd:
        subprocess.run(cmd, shell=True, check=True, stdout=log_fd,
                       stderr=subprocess.STDOUT)
    else:
      print(f'Skipping {name}')

  print('Creating CSV')
  with open(csv_file, 'w') as csv:
    print('name,batch_size,batch_time,sample_rate,ceff,seff,teff,mem1,mem2,'
          'tp,pp,dp,tn,pn,dn,pi,mbs,recompute,tp_comm,redo,w_off,a_off,'
          'o_off,tp_time,pp_time,dp_exp,fw_off_exp,bw_off_exp,total,good,bad',
          file=csv)
    for config, nodes in configs:
      assert os.path.isfile(config), f'"{config}" does not exist'
      name = os.path.splitext(os.path.basename(config))[0]
      log = os.path.join(args.directory, f'{name}.log')
      exe = os.path.join(args.directory, f'{name}_exe.json')
      stats = os.path.join(args.directory, f'{name}_stats.json')
      raw = os.path.join(args.directory, f'{name}_raw.json')

      if args.verbose:
        print(f'  Opening {exe}')
      with open(exe, 'r') as fd:
        e = json.load(fd)
      if args.verbose:
        print(f'  Opening {stats}')
      with open(stats, 'r') as fd:
        s = json.load(fd)

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
      batch_time = s['total_time']
      sample_rate = batch_size / batch_time
      ceff = s['compute_efficiency']
      seff = s['system_efficiency']
      teff = s['total_efficiency']
      mem1 = s['proc_mem_tier1_cap_req'] / 1024**3
      mem2 = s['proc_mem_tier2_cap_req'] / 1024**3
      woff = e['weight_offload']
      aoff = e['activations_offload']
      ooff = e['optimizer_offload']
      tp_time = s['tp_comm_time']
      pp_time = s['pp_comm_time']
      dp_exp = s['dp_exposed_comm_time']
      fw_off_exp = s['fw_offload_exposed_time']
      bw_off_exp = s['fw_offload_exposed_time']
      total, good, bad = get_executions(log, args.verbose)
      print(f'{name},{batch_size},{batch_time},{sample_rate},{ceff},{seff},'
            f'{teff},{mem1},{mem2},{tp},{pp},{dp},{tn},{pn},{dn},'
            f'{pi},{mbs},{ar},{tp_comm},{redo},{woff},{aoff},{ooff},{tp_time},'
            f'{pp_time},{dp_exp},{fw_off_exp},{bw_off_exp},{total},{good},'
            f'{bad}',
            file=csv)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('application', type=str,
                  help='Application configuration')
  ap.add_argument('configs', type=str,
                  help='Path to configs.json')
  ap.add_argument('batch_mode', type=int,
                  help='0 for num_procs, >0 for max')
  ap.add_argument('directory', type=str,
                  help='Output directory')
  ap.add_argument('-v', '--verbose', action='store_true',
                  help='Verbose output')
  ap.add_argument('-c', '--csvonly', action='store_true',
                  help='Only create CSVs')
  args = ap.parse_args()
  main(args)
