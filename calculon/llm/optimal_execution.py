"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  https://www.apache.org/licenses/LICENSE-2.0
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

import datetime
import gzip
import logging
import multiprocessing as mp
import psutil
import os

import calculon
from calculon.util import pick, arg_true_false_all
from calculon.llm import *


class OptimalExecution(calculon.CommandLine):
  NAME = 'llm-optimal-execution'
  ALIASES = ['loe']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      OptimalExecution.NAME, aliases=OptimalExecution.ALIASES,
      help='run a search to find the optimal llm execution')
    sp.set_defaults(func=OptimalExecution.run_command)
    sp.add_argument('-d', '--debug', action='store_true',
                    help='Loop over executions, don\'t run them')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('num_procs', type=int,
                    help='Number of processors in execution')
    sp.add_argument('max_batch_size', type=int,
                    help='Maximum batch size, will be largest multiple of DP')
    sp.add_argument('datatype', type=str, choices=System.supported_datatypes(),
                    help='The datatype to use')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('output', type=str,
                    help='File path to the output file'
                    " ('*.csv', '*.csv.gz', '*.json', '*.json.gz')")
    sp.add_argument('-c', '--cpus', type=int, default=psutil.cpu_count(logical=False),
                    help='CPUs to use for parallelization')
    sp.add_argument('-n', '--noneok', action='store_true',
                    help='Don\'t give failure status when no good execution exists')
    sp.add_argument('-m', '--mbs-break', action='store_true',
                    help='Search across MBS and break earlier when possible')
    sp.add_argument('-t', '--top-n', type=int, default=1,
                    help='Number of best outputs')
    sp.add_argument('-l', '--layers', action='store_true',
                    help='Include layers information in output stats file')
    sp.add_argument('-f', '--fused_activation', type=arg_true_false_all,
                    default='true', help='Mode of fused activation')
    sp.add_argument('--no-tp-overlap', action='store_true',
                    help='Don\'t allow TP overlap')
    sp.add_argument('--no-dp-overlap', action='store_true',
                    help='Don\'t allow DP overlap')

  @staticmethod
  def run_command(logger, args):
    assert args.top_n > 0, 'top-n must be > 0'

    app = Llm.Application(calculon.io.read_json_file(args.application))
    syst = System(calculon.io.read_json_file(args.system))

    params = []
    for tp in Llm.get_all_tensor_parallelisms(
        args.num_procs, app.hidden, app.attn_heads):
      for pp in Llm.get_all_pipeline_parallelisms(
          args.num_procs, tp, app.num_blocks):
        dp = Llm.get_data_parallelism(args.num_procs, tp, pp)
        for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
          batch_size = OptimalExecution.get_batch_size(dp, args.max_batch_size)
          if batch_size is None:
            continue
          for activation_recompute in ['full', 'attn_only', 'none']:
            for optimizer_sharding in pick(dp>1, [True, False], [False]):
              for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
                params.append(
                  (args.debug, args.top_n, args.layers, args.num_procs,
                   args.max_batch_size, args.datatype, app, syst, tp, pp, dp,
                   ppint, batch_size, activation_recompute, optimizer_sharding,
                   tensor_par_comm_type, args.fused_activation, args.mbs_break,
                   not args.no_tp_overlap, not args.no_dp_overlap))

    # Runs parallel searches
    start_time = datetime.datetime.now()
    with mp.Pool(args.cpus) as pool:
      searches = pool.starmap(OptimalExecution.search, params)
    end_time = datetime.datetime.now()

    # Combines parallel search result into one data structure
    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    for cbest, ec, gec, bec, tp, pp in searches:
      best = OptimalExecution.update_list(best, cbest, args.top_n)
      exe_count += ec
      good_exe_count += gec
      bad_exe_count += bec

    logger.info(f'Total executions: {exe_count}')
    logger.info(f'Good executions: {good_exe_count}')
    logger.info(f'Bad executions: {bad_exe_count}')
    calc_rate = exe_count / (end_time - start_time).total_seconds()
    logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
    if args.debug:
      return 0

    if len(best) == 0:
      if not args.noneok:
        logger.fatal('No acceptable configurations found :(')
        return -1
      else:
        logger.info('No acceptable configurations found :(')
    else:
      logger.info(f'Best sample rate: {best[0][0]}')

    output = {}
    for index, run in enumerate(best):
      _, execution, stats = run
      output[index] = {
        'execution': execution,
        'stats': stats
      }

    if calculon.io.is_json_extension(args.output):
      logger.info(f'Output: {args.output}')
      calculon.io.write_json_file(output, args.output)
    elif args.output.endswith('.csv') or args.output.endswith('.csv.gz'):
      logger.info(f'Output: {args.output}')
      exe_keys = list(output[0]['execution'].keys())
      stats_keys = list(output[0]['stats'].keys())
      opener = gzip.open if args.output.endswith('.gz') else open
      with opener(args.output, 'wb') as fd:
        fd.write(bytes(f',{",".join(exe_keys)},{",".join(stats_keys)}\n',
                       'utf-8'))
        for index in sorted(output.keys()):
          fd.write(bytes(f'{index}', 'utf-8'))
          for exe_key in exe_keys:
            fd.write(bytes(f',{output[index]["execution"][exe_key]}', 'utf-8'))
          for stats_key in stats_keys:
            fd.write(bytes(f',{output[index]["stats"][stats_key]}', 'utf-8'))
          fd.write(bytes('\n', 'utf-8'))
    else:
      assert False, f'Unknown file type: {args.output}'

    return 0

  @staticmethod
  def get_batch_size(data_par, max_batch_size):
    if data_par > max_batch_size:
      return None
    last = data_par
    while True:
      if last + data_par > max_batch_size:
        return last
      else:
        last += data_par

  @staticmethod
  def search(debug, top_n, layers, num_procs, max_batch_size, datatype,
             app, syst, tp, pp, dp, ppint, batch_size, activation_recompute,
             optimizer_sharding, tensor_par_comm_type, fused_acts, mbs_break,
             allow_tp_overlap, allow_dp_overlap):
    num_nets = syst.num_networks

    best = []
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0

    has_mem2 = syst.mem2.capacity > 0

    can_redo = Llm.can_redo_ag(tensor_par_comm_type,
                               activation_recompute)
    for seq_par_ag_redo in pick(can_redo, [True, False], [False]):
      for data_par_overlap in pick(dp>1 and allow_dp_overlap, [True, False],
                                   [False]):
        for tensor_par_overlap in pick(tp>1 and allow_tp_overlap,
                                       ['none', 'ring', 'pipe'], ['none']):
          for weight_offload in pick(has_mem2, [True, False], [False]):
            if activation_recompute == 'full' or not has_mem2:
              activations_offloads = [False]
            else:
              activations_offloads = [True, False]
            for activations_offload in activations_offloads:
              for optimizer_offload in pick(has_mem2, [True, False],
                                            [False]):
                for fused_act in fused_acts:
                  for microbatch_size in Llm.get_valid_microbatch_sizes(
                      app.seq_size, tp, dp, batch_size, pp):
                    mbs_break_good = good_exe_count
                    for tn in pick(tp>1, range(num_nets), [0]):
                      for pn in pick(pp>1, range(num_nets), [0]):
                        for dn in pick(dp>1, range(num_nets), [0]):
                          exe_count += 1
                          exe_json = {
                            'num_procs': num_procs,
                            'tensor_par': tp,
                            'pipeline_par': pp,
                            'data_par': dp,
                            'tensor_par_net': tn,
                            'pipeline_par_net': pn,
                            'data_par_net': dn,
                            'batch_size': batch_size,
                            'microbatch_size': microbatch_size,
                            'datatype': datatype,
                            'fused_activation': fused_act,
                            'attention_type': 'multihead',
                            'activation_recompute': activation_recompute,
                            'pipeline_interleaving': ppint,
                            'optimizer_sharding': optimizer_sharding,
                            'tensor_par_comm_type': tensor_par_comm_type,
                            'tensor_par_overlap': tensor_par_overlap,
                            'seq_par_ag_redo': seq_par_ag_redo,
                            'data_par_overlap': data_par_overlap,
                            'weight_offload': weight_offload,
                            'activations_offload': activations_offload,
                            'optimizer_offload': optimizer_offload,
                            'training': True
                          }

                          if not debug:
                            try:
                              logger = logging.Logger('sub')
                              model = Llm(app, logger)
                              model.compile(
                                syst,
                                Llm.Execution.from_json(exe_json))
                              model.run(syst)
                              stats = model.get_stats_json(layers)
                              good_exe_count += 1
                              curr = (stats['sample_rate'], exe_json, stats)
                              best = OptimalExecution.update_list(best, curr,
                                                                  top_n)
                            except Llm.Error as ex:
                              logger = logging.getLogger()
                              logger.debug(f'JSON:{exe_json}\nERROR:{ex}\n')
                              bad_exe_count += 1
                    if mbs_break and good_exe_count == mbs_break_good:
                      break
    return (best, exe_count, good_exe_count, bad_exe_count, tp, pp)

  @staticmethod
  def update_list(current, candidate, quantity):
    if not isinstance(candidate, list):
      current.append(candidate)
    else:
      current.extend(candidate)
    current.sort(reverse=True, key=lambda x: x[0])
    return current[:quantity]


calculon.CommandLine.register(OptimalExecution)
