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
import json
import logging
import multiprocessing as mp
import os

import calculon
from calculon.util import pick
from calculon.megatron import *

def search(debug, num_procs, app, syst, tp, pp):
  best_time = None
  best_stats = None
  best_exe = None
  exe_count = 0
  good_exe_count = 0
  bad_exe_count = 0

  dp = Megatron.get_data_parallelism(num_procs, tp, pp)
  for ppint in Megatron.get_valid_pipeline_interleavings(app.num_blocks, pp):
    batch_size = num_procs
    assert batch_size % dp == 0
    for microbatch_size in Megatron.get_valid_microbatch_sizes(app.seq_size, tp, dp, batch_size, pp):
      for activation_recompute in ['full', 'attn_only', 'none']:
        for optimizer_sharding in pick(dp>1, [True, False], [False]):
          for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
            can_redo = Megatron.can_redo_ag(tensor_par_comm_type, activation_recompute)
            for seq_par_ag_redo in pick(can_redo, [True, False], [False]):
              for data_par_overlap in pick(dp>1, [True, False], [False]):
                for weight_offload in [True, False]:
                  if activation_recompute == 'full':
                    activations_offloads = [False]
                  else:
                    activations_offloads = [True, False]
                  for activations_offload in activations_offloads:
                    for optimizer_offload in [True, False]:
                      exe_count += 1
                      exe_json = {
                        'num_procs': num_procs,
                        'tensor_par': tp,
                        'pipeline_par': pp,
                        'data_par': dp,
                        'batch_size': batch_size,
                        'microbatch_size': microbatch_size,
                        'datatype': 'bfloat16',
                        'activation_recompute': activation_recompute,
                        'pipeline_interleaving': ppint,
                        'optimizer_sharding': optimizer_sharding,
                        'tensor_par_comm_type': tensor_par_comm_type,
                        'seq_par_ag_redo': seq_par_ag_redo,
                        'data_par_overlap': data_par_overlap,
                        'weight_offload': weight_offload,
                        'activations_offload': activations_offload,
                        'optimizer_offload': optimizer_offload,
                        'training': True
                      }

                      if not debug:
                        try:
                          logger = logging.getLogger()
                          model = Megatron(app, logger)
                          model.compile(Megatron.Execution(exe_json))
                          model.run(syst)
                          stats = model.get_json()
                          good_exe_count += 1
                          if best_time == None or stats['total_time'] < best_time:
                            best_time = stats['total_time']
                            best_exe = exe_json
                            best_stats = stats
                        except Megatron.Error as ex:
                          bad_exe_count += 1
  return (best_time, best_stats, best_exe, exe_count, good_exe_count,
          bad_exe_count, tp, pp)


class OptimalExecution(calculon.CommandLine):
  NAME = 'megatron-optimal-execution'
  ALIASES = ['moe']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      OptimalExecution.NAME, aliases=OptimalExecution.ALIASES,
      help='run a search to find the optimal megatron execution')
    sp.set_defaults(func=OptimalExecution.run_command)
    sp.add_argument('-d', '--debug', action='store_true',
                    help='Loop over executions, don\'t run them')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('num_procs', type=int,
                    help='Number of processors in execution')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('-e', '--execution', type=str, default=None,
                    help='File path to execution output')
    sp.add_argument('-s', '--stats', type=str, default=None,
                    help='File path to stats output')
    sp.add_argument('-r', '--raw', type=str, default=None,
                    help='File path to raw TP/PP output')
    sp.add_argument('-c', '--cpus', type=int, default=os.cpu_count(),
                    help='CPUs to use for parallelization')

  @staticmethod
  def run_command(logger, args):
    with open(args.application, 'r') as fd:
      app = Megatron.Application(json.load(fd))
    with open(args.system, 'r') as fd:
      syst = System(json.load(fd))

    params = []
    for tp in Megatron.get_all_tensor_parallelisms(args.num_procs, app.attn_heads):
      for pp in Megatron.get_all_pipeline_parallelisms(args.num_procs, tp, app.num_blocks):
        params.append((args.debug, args.num_procs, app, syst, tp, pp))

    start_time = datetime.datetime.now()
    with mp.Pool(args.cpus) as pool:
      searches = pool.starmap(search, params)
    end_time = datetime.datetime.now()

    best_time = None
    best_stats = None
    best_exe = None
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    data = {}
    for bt, bs, be, ec, gec, bec, tp, pp in searches:
      if best_time == None or (bt != None and bt < best_time):
        best_time = bt
        best_stats = bs
        best_exe = be
      exe_count += ec
      good_exe_count += gec
      bad_exe_count += bec
      if tp not in data:
        data[tp] = {}
      if bt != None:
        data[tp][pp] = bt
      else:
        data[tp][pp] = float('NaN')

    logger.info(f'Total executions: {exe_count}')
    logger.info(f'Good executions: {good_exe_count}')
    logger.info(f'Bad executions: {bad_exe_count}')
    calc_rate = exe_count / (end_time - start_time).total_seconds()
    logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
    if not args.debug:
      if not best_time:
        logger.fatal('No acceptable configurations found :(')
        return -1
      else:
        logger.info(f'Best total time: {best_time}')
      if args.execution:
        with open(args.execution, 'w') as fd:
          json.dump(best_exe, fd, indent=2)
        logger.info(f'Best execution: {args.execution}')
      if args.stats:
        with open(args.stats, 'w') as fd:
          json.dump(best_stats, fd, indent=2)
        logger.info(f'Best stats: {args.stats}')
      if args.raw:
        with open(args.raw, 'w') as fd:
          json.dump(data, fd, indent=2, allow_nan=True)
        logger.info(f'Raw TP/PP: {args.raw}')

    return 0

calculon.CommandLine.register(OptimalExecution)
