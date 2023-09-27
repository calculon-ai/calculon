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
import itertools
import logging
import math
import multiprocessing as mp
import os
import pandas
import psutil
import random

import calculon
from calculon.util import pick, arg_true_false_all
from calculon.llm import *


class AllExecutions(calculon.CommandLine):
  NAME = 'llm-all-executions'
  ALIASES = ['lae']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      AllExecutions.NAME, aliases=AllExecutions.ALIASES,
      help='run a search to find the optimal llm execution')
    sp.set_defaults(func=AllExecutions.run_command)
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
                    " ('*.csv', '*.csv.gz')")
    sp.add_argument('-c', '--cpus', type=int, default=psutil.cpu_count(logical=False),
                    help='CPUs to use for parallelization')
    sp.add_argument('-n', '--noneok', action='store_true',
                    help='Don\'t give failure status when no good execution exists')
    sp.add_argument('-f', '--fused_activation', type=arg_true_false_all,
                    default='true', help='Mode of fused activation')

  @staticmethod
  def execution_fields():
    return (
      'num_procs', 'tensor_par', 'pipeline_par', 'data_par', 'tensor_par_net',
      'pipeline_par_net', 'data_par_net', 'batch_size', 'microbatch_size',
      'datatype', 'fused_activation', 'attention_type', 'activation_recompute',
      'pipeline_interleaving', 'optimizer_sharding', 'tensor_par_comm_type',
      'tensor_par_overlap', 'seq_par_ag_redo', 'data_par_overlap',
      'weight_offload', 'activations_offload', 'optimizer_offload', 'training')

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
  def all_executions(app, syst, num_procs, max_batch_size, datatype, fused_activation):
    has_mem2 = syst.mem2.capacity > 0
    num_nets = syst.num_networks
    count = 0
    for tp in Llm.get_all_tensor_parallelisms(
        num_procs, app.hidden, app.attn_heads):
      for pp in Llm.get_all_pipeline_parallelisms(
          num_procs, tp, app.num_blocks):
        dp = Llm.get_data_parallelism(num_procs, tp, pp)
        for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
          batch_size = AllExecutions.get_batch_size(dp, max_batch_size)
          if batch_size is None:
            continue
          for activation_recompute in ['full', 'attn_only', 'none']:
            for optimizer_sharding in pick(dp>1, [True, False], [False]):
              for tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']:
                can_redo = Llm.can_redo_ag(tensor_par_comm_type,
                                           activation_recompute)
                for seq_par_ag_redo in pick(can_redo, [True, False], [False]):
                  for data_par_overlap in pick(dp>1, [True, False], [False]):
                    for tensor_par_overlap in pick(tp>1, ['none', 'ring', 'pipe'], ['none']):
                      for weight_offload in pick(has_mem2, [True, False], [False]):
                        if activation_recompute == 'full' or not has_mem2:
                          activations_offloads = [False]
                        else:
                          activations_offloads = [True, False]
                        for activations_offload in activations_offloads:
                          for optimizer_offload in pick(has_mem2, [True, False],
                                                        [False]):
                            for fused_act in fused_activation:
                              for microbatch_size in Llm.get_valid_microbatch_sizes(
                                  app.seq_size, tp, dp, batch_size, pp):
                                for tn in pick(tp>1, range(num_nets), [0]):
                                  for pn in pick(pp>1, range(num_nets), [0]):
                                    for dn in pick(dp>1, range(num_nets), [0]):
                                      yield (num_procs, tp, pp, dp, tn, pn, dn,
                                             batch_size, microbatch_size, datatype,
                                             fused_act, 'multihead', activation_recompute,
                                             ppint, optimizer_sharding, tensor_par_comm_type,
                                             tensor_par_overlap, seq_par_ag_redo,
                                             data_par_overlap, weight_offload,
                                             activations_offload, optimizer_offload,
                                             True)
                                      count += 1

  @staticmethod
  def run_command(logger, args):
    assert args.output.endswith('.csv') or args.output.endswith('.csv.gz')

    app = Llm.Application(calculon.io.read_json_file(args.application))
    syst = System(calculon.io.read_json_file(args.system))

    executions = list(AllExecutions.all_executions(
      app, syst, args.num_procs, args.max_batch_size, args.datatype,
      args.fused_activation))
    random.shuffle(executions)
    exe_count = len(executions)
    logger.info(f'Total executions: {exe_count}')

    step = math.ceil(len(executions) / args.cpus)
    worker_args = []
    for index in range(0, len(executions), step):
      worker_args.append((app, syst, executions[index : index + step]))
    del executions

    # Runs parallel searches
    start_time = datetime.datetime.now()
    with mp.Pool(args.cpus) as pool:
      goods = pool.starmap(AllExecutions.search, worker_args)
    end_time = datetime.datetime.now()
    good_count = sum(len(good) for good in goods)

    # Console statistics
    logger.info(f'Good executions: {good_count}')
    logger.info(f'Bad executions: {exe_count-good_count}')
    calc_rate = exe_count / (end_time - start_time).total_seconds()
    logger.info(f'Calculation rate: {calc_rate:.2f} calcs/sec')

    # Check if OK
    if good_count == 0:
      if not args.noneok:
        logger.fatal('No acceptable configurations found :(')
        return -1
      else:
        logger.info('No acceptable configurations found :(')

    if args.debug:
      return 0

    # Writes to CSV
    fields = Llm.Execution.fields() + Llm.get_stats_fields()
    assert len(fields) == len(goods[0][0])
    logger.info(f'Output: {args.output}')
    opener = gzip.open if args.output.endswith('.gz') else open
    with opener(args.output, 'wb') as fd:
      fd.write(bytes(','.join(fields) + '\n', 'utf-8'))
      for vals in itertools.chain(*goods):
        fd.write(bytes(','.join(str(v) for v in vals) + '\n', 'utf-8'))

    return 0

  @staticmethod
  def search(app, syst, executions):
    good = []
    for execution in executions:
      try:
        model = Llm(app, logging.Logger('sub'))
        model.compile(syst, Llm.Execution(*execution))
        model.run(syst)
        statistics = model.get_stats_values()
        good.append(execution + statistics)
      except Llm.Error as ex:
        logger = logging.getLogger()
        logger.debug(f'ERROR:{ex}\n')
    return good

  @staticmethod
  def update_list(current, candidate, quantity):
    if not isinstance(candidate, list):
      current.append(candidate)
    else:
      current.extend(candidate)
    if quantity <= 0:
      return current  # don't sort and chop
    else:
      current.sort(reverse=True, key=lambda x: x[0])
      return current[:quantity]


calculon.CommandLine.register(AllExecutions)
