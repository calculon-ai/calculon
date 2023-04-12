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
import math
import multiprocessing as mp
import psutil
import random
import os

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
    sp.add_argument('-e', '--execution', type=str, default=None,
                    help='File path to execution output')
    sp.add_argument('-s', '--stats', type=str, default=None,
                    help='File path to stats output')
    sp.add_argument('-c', '--cpus', type=int, default=psutil.cpu_count(logical=False),
                    help='CPUs to use for parallelization')
    sp.add_argument('-n', '--noneok', action='store_true',
                    help='Don\'t give failure status when no good execution exists')

  @staticmethod
  def run_command(logger, args):
    with open(args.application, 'r') as fd:
      app = Megatron.Application(json.load(fd))
    with open(args.system, 'r') as fd:
      syst = System(json.load(fd))

    print('Creating execution list')
    all_executions = OptimalExecution.create_executions(
      args.num_procs, args.max_batch_size, args.datatype, app, syst)
    print(f'{len(all_executions)} executions')

    print('Shuffling execution list')
    random.shuffle(all_executions)

    print('Assigning execution ranges to CPUs')
    step = math.ceil(len(all_executions) / args.cpus)
    start = 0
    params = []
    for cpu in range(args.cpus):
      end = min(start + step, len(all_executions) - 1)
      num = end - start + 1
      params.append((app, syst, all_executions, start, num))
      start += num

    print('Running search')
    start_time = datetime.datetime.now()
    with mp.Pool(args.cpus) as pool:
      searches = pool.starmap(OptimalExecution.search, params)
    end_time = datetime.datetime.now()

    print('Analyzing search results')
    best_rate = None
    best_stats = None
    best_exe = None
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0
    for br, bs, be, ec, gec, bec in searches:
      if best_rate == None or (br != None and br > best_rate):
        best_rate = br
        best_stats = bs
        best_exe = be
      exe_count += ec
      good_exe_count += gec
      bad_exe_count += bec

    logger.info(f'Total executions: {exe_count}')
    logger.info(f'Good executions: {good_exe_count}')
    logger.info(f'Bad executions: {bad_exe_count}')
    calc_rate = exe_count / (end_time - start_time).total_seconds()
    logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')
    if not args.debug:
      none_found = not best_rate
      if none_found:
        if not args.noneok:
          logger.fatal('No acceptable configurations found :(')
          return -1
        else:
          logger.info('No acceptable configurations found :(')
      else:
        logger.info(f'Best sample rate: {best_rate}')
      if args.execution:
        with open(args.execution, 'w') as fd:
          json.dump({} if none_found else best_exe, fd, indent=2)
        logger.info(f'Best execution: {args.execution}')
      if args.stats:
        with open(args.stats, 'w') as fd:
          json.dump({} if none_found else best_stats, fd, indent=2)
        logger.info(f'Best stats: {args.stats}')
    return 0

  @staticmethod
  def get_batch_size(data_par, max_batch_size):
    last = data_par
    while True:
      if last + data_par > max_batch_size:
        return last
      else:
        last += data_par

  @staticmethod
  def create_executions(num_procs, max_batch_size, datatype, app, syst):
    all_executions = []

    num_nets = syst.num_networks
    has_mem2 = syst.mem2.capacity > 0

    for tp in Megatron.get_all_tensor_parallelisms(num_procs, app.hidden, app.attn_heads):
      for pp in Megatron.get_all_pipeline_parallelisms(num_procs, tp, app.num_blocks):
        dp = Megatron.get_data_parallelism(num_procs, tp, pp)
        for ppint in Megatron.get_valid_pipeline_interleavings(app.num_blocks, pp):
          batch_size = OptimalExecution.get_batch_size(dp, max_batch_size)
          assert batch_size % dp == 0
          for microbatch_size in Megatron.get_valid_microbatch_sizes(
              app.seq_size, tp, dp, batch_size, pp):
            for activation_recompute in ['full', 'attn_only', 'none']:
              for optimizer_sharding in pick(dp>1, [True, False], [False]):
                for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
                  can_redo = Megatron.can_redo_ag(tensor_par_comm_type,
                                                  activation_recompute)
                  for seq_par_ag_redo in pick(can_redo, [True, False], [False]):
                    for data_par_overlap in pick(dp>1, [True, False], [False]):
                      for tensor_par_overlap in pick(tp>1, [True, False], [False]):
                        for weight_offload in pick(has_mem2, [True, False], [False]):
                          if activation_recompute == 'full' or not has_mem2:
                            activations_offloads = [False]
                          else:
                            activations_offloads = [True, False]
                          for activations_offload in activations_offloads:
                            for optimizer_offload in pick(has_mem2, [True, False],
                                                          [False]):
                              for tn in pick(tp>1, range(num_nets), [0]):
                                for pn in pick(pp>1, range(num_nets), [0]):
                                  for dn in pick(dp>1, range(num_nets), [0]):
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
                                      'fused_activation': True,
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
                                    all_executions.append(exe_json)
    return all_executions

  @staticmethod
  def search(app, syst, all_executions, start, num):
    best_rate = None
    best_stats = None
    best_exe = None
    good_exe_count = 0
    bad_exe_count = 0
    for exe_idx in range(start, start + num):
      try:
        exe_json = all_executions[exe_idx]
      except IndexError as ex:
        print(f'idx={exe_idx}')
        raise ex
      try:
        logger = logging.Logger('sub')
        model = Megatron(app, logger)
        model.compile(syst, Megatron.Execution(exe_json))
        model.run(syst)
        stats = model.get_stats_json()
        good_exe_count += 1
        if (best_rate == None or
            stats['sample_rate'] > best_rate):
          best_rate = stats['sample_rate']
          best_exe = exe_json
          best_stats = stats
      except Megatron.Error as ex:
        logger = logging.getLogger()
        logger.debug(f'JSON:{exe_json}\nERROR:{ex}\n')
        bad_exe_count += 1
    assert good_exe_count + bad_exe_count == num
    return (best_rate, best_stats, best_exe, num, good_exe_count,
            bad_exe_count)


calculon.CommandLine.register(OptimalExecution)
