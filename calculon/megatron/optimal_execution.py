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

import calculon
from calculon.util import pick
from calculon.megatron import *

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
    sp.add_argument('-n', '--nofail', action='store_true',
                    help='Ensure that no calculation fails')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('num_procs', type=int,
                    help='Number of processors in execution')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('execution', type=str,
                    help='File path to execution output')
    sp.add_argument('stats', type=str,
                    help='File path to stats output')

  @staticmethod
  def run_command(logger, args):
    with open(args.application, 'r') as fd:
      app = Megatron.Application(json.load(fd))
    with open(args.system, 'r') as fd:
      syst = System(json.load(fd))

    best_time = None
    best_stats = None
    best_exe = None
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    start_time = datetime.datetime.now()
    for tp in Megatron.get_all_tensor_parallelisms(args.num_procs, app.attn_heads):
      for pp in Megatron.get_all_pipeline_parallelisms(args.num_procs, tp, app.num_blocks):
        dp = Megatron.get_data_parallelism(args.num_procs, tp, pp)
        for ppint in Megatron.get_valid_pipeline_interleavings(app.num_blocks, pp):
          batch_size = 4096
          assert batch_size % dp == 0
          for minibatch_size in Megatron.get_valid_minibatch_sizes(dp, batch_size, pp):
            for activation_recompute in ['full', 'partial', 'none']:
              for optimizer_sharding in pick(dp>1, [True, False], [False]):
                for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
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
                            'num_procs': args.num_procs,
                            'tensor_par': tp,
                            'pipeline_par': pp,
                            'data_par': dp,
                            'batch_size': batch_size,
                            'minibatch_size': minibatch_size,
                            'datatype': 'bfloat16',
                            'activation_recompute': activation_recompute,
                            'pipeline_interleaving': ppint,
                            'optimizer_sharding': optimizer_sharding,
                            'tensor_par_comm_type': tensor_par_comm_type,
                            'data_par_overlap': data_par_overlap,
                            'weight_offload': weight_offload,
                            'activations_offload': activations_offload,
                            'optimizer_offload': optimizer_offload,
                            'training': True
                          }
                          if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'{json.dumps(exe_json, indent=2)}')

                          if not args.debug:
                            try:
                              model = Megatron(app, logger)
                              model.compile(Megatron.Execution(exe_json))
                              model.run(syst)
                              stats = model.get_json()
                              logger.info(f'{exe_count} -> {stats["total_time"]}')
                              good_exe_count += 1
                              if best_time == None or stats['total_time'] < best_time:
                                best_time = stats['total_time']
                                best_exe = exe_json
                                best_stats = stats
                            except Megatron.Error as ex:
                              logger.info(f'{exe_count} -> {ex}')
                              bad_exe_count += 1
                              if args.nofail:
                                return -1
                          else:
                            logger.info(f'{exe_count}')

    end_time = datetime.datetime.now()

    if not args.debug:
      logger.info(f'Total executions: {exe_count}')
      logger.info(f'Good executions: {good_exe_count}')
      logger.info(f'Bad executions: {bad_exe_count}')
      calc_rate = exe_count / (end_time - start_time).total_seconds()
      logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
      if not best_time:
        logger.fatal('No acceptable configurations found :(')
        return -1
      logger.info(f'Best total time: {best_time}')
      with open(args.execution, 'w') as fd:
        json.dump(best_exe, fd, indent=2)
      logger.info(f'Best execution: {args.execution}')
      with open(args.stats, 'w') as fd:
        json.dump(best_stats, fd, indent=2)
      logger.info(f'Best stats: {args.stats}')

    return 0

calculon.CommandLine.register(OptimalExecution)
