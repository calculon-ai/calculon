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
import os

import calculon
from calculon.util import pick
from calculon.llm import *


CSV_EXE_FIELDS = [
  'num_procs', 'tensor_par', 'pipeline_par', 'data_par', 'tensor_par_net',
  'pipeline_par_net', 'data_par_net', 'batch_size', 'microbatch_size',
  'activation_recompute', 'pipeline_interleaving', 'optimizer_sharding',
  'tensor_par_comm_type', 'tensor_par_overlap', 'seq_par_ag_redo',
  'data_par_overlap', 'weight_offload', 'activations_offload',
  'optimizer_offload']
CSV_STATS_FIELDS = [
  'block_fw_flops', 'block_fw_flops_time', 'block_fw_mem_accessed',
  'block_fw_mem_time', 'block_fw_time', 'baseblock_fw_tp_time',
  'edgeblock_fw_tp_time', 'baseblock_fw_tp_time_exposed',
  'edgeblock_fw_tp_time_exposed', 'block_re_flops', 'block_re_flops_time',
  'block_re_mem_accessed', 'block_re_mem_time', 'block_re_time',
  'baseblock_recomm_time', 'edgeblock_recomm_time',
  'baseblock_recomm_time_exposed', 'edgeblock_recomm_time_exposed',
  'block_agrad_flops', 'block_agrad_flops_time', 'block_agrad_mem_accessed',
  'block_agrad_mem_time', 'block_agrad_time', 'baseblock_agrad_tp_time',
  'edgeblock_agrad_tp_time', 'baseblock_agrad_tp_time_exposed',
  'edgeblock_agrad_tp_time_exposed', 'block_wgrad_flops',
  'block_wgrad_flops_time', 'block_wgrad_mem_accessed', 'block_wgrad_mem_time',
  'block_wgrad_time', 'baseblock_wgrad_tp_time', 'edgeblock_wgrad_tp_time',
  'baseblock_wgrad_tp_time_exposed', 'edgeblock_wgrad_tp_time_exposed',
  'block_optim_flops', 'block_optim_flops_time', 'block_optim_mem_accessed',
  'block_optim_mem_time', 'block_optim_time', 'block_fw_tp_size',
  'block_bw_tp_size', 'block_recomm_size', 'block_fw_pp_size',
  'block_bw_pp_size', 'block_dp_size', 'block_weight_space',
  'block_act_working_space', 'block_act_storage_space',
  'block_act_checkpoint_size', 'block_weight_grad_space',
  'block_weight_grad_space_no_sharding', 'block_act_grad_space',
  'block_optimizer_space', 'weight_space', 'act_space', 'act_checkpoint_size',
  'act_grad_space', 'weight_grad_space', 'optimizer_space', 'fw_time',
  'bw_time', 'optim_step_time', 'recompute_time', 'recomm_link_time',
  'recomm_exposed_time', 'bubble_time', 'tp_comm_link_time',
  'pp_comm_link_time', 'dp_comm_link_time', 'tp_comm_exposed_time',
  'pp_comm_exposed_time', 'dp_comm_exposed_time', 'fw_offload_exposed_time',
  'bw_offload_exposed_time', 'total_time', 'act_offload_bw_req',
  'weight_offload_bw_req', 'optim_offload_bw_req', 'offload_mem_bw_req',
  'proc_mem_tier1_cap_req', 'proc_mem_tier2_cap_req', 'useful_flops',
  'compute_efficiency', 'system_efficiency', 'total_efficiency', 'sample_rate']


def get_batch_size(data_par, max_batch_size):
  last = data_par
  while True:
    if last + data_par > max_batch_size:
      return last
    else:
      last += data_par


def write_csv_header(csv):
  print('#,', file=csv, end='')
  for field in CSV_EXE_FIELDS:
    print(f'{field},', file=csv, end='')
  for field in CSV_STATS_FIELDS:
    print(f'{field},', file=csv, end='')
  print('', file=csv)


def write_csv_data(csv, num, exe, stats):
  print(f'{num},', file=csv, end='')
  for field in CSV_EXE_FIELDS:
    print(f'{exe[field]},', file=csv, end='')
  for field in CSV_STATS_FIELDS:
    print(f'{stats[field]},', file=csv, end='')
  print('', file=csv)


class AllExecutions(calculon.CommandLine):
  NAME = 'llm-all-executions'
  ALIASES = ['lae']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      AllExecutions.NAME, aliases=AllExecutions.ALIASES,
      help='run a search to find the result of all llm executions')
    sp.set_defaults(func=AllExecutions.run_command)
    sp.add_argument('-d', '--debug', action='store_true',
                    help='Loop over executions, don\'t run them')
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('num_procs', type=int,
                    help='Number of processors in execution')
    sp.add_argument('max_batch_size', type=int,
                    help='Maximum batch size, will be largest multiple of DP')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('-e', '--execution', type=str, default=None,
                    help='File path to execution output')
    sp.add_argument('-s', '--stats', type=str, default=None,
                    help='File path to stats output')
    sp.add_argument('-c', '--csv', type=str, default=None,
                    help='File path to CSV output file')
    sp.add_argument('-n', '--noneok', action='store_true',
                    help='Don\'t give failure status when no good execution exists')

  @staticmethod
  def run_command(logger, args):
    with open(args.application, 'r') as fd:
      app = Llm.Application(json.load(fd))
    with open(args.system, 'r') as fd:
      syst = System(json.load(fd))

    best_rate = None
    best_stats = None
    best_exe = None
    exe_count = 0
    good_exe_count = 0
    bad_exe_count = 0

    num_nets = syst.num_networks
    has_mem2 = syst.mem2.capacity > 0

    start_time = datetime.datetime.now()

    if args.csv:
      csv = open(args.csv, 'w')
      write_csv_header(csv)

    for tp in Llm.get_all_tensor_parallelisms(
        args.num_procs, app.attn_heads):
      for pp in Llm.get_all_pipeline_parallelisms(
          args.num_procs, tp, app.num_blocks):
        dp = Llm.get_data_parallelism(args.num_procs, tp, pp)
        for ppint in Llm.get_valid_pipeline_interleavings(app.num_blocks, pp):
          batch_size = get_batch_size(dp, args.max_batch_size)
          assert batch_size % dp == 0
          for microbatch_size in Llm.get_valid_microbatch_sizes(
              app.seq_size, tp, dp, batch_size, pp):
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
                            for optimizer_offload in pick(has_mem2, [True, False], [False]):
                              for tn in pick(tp>1, range(num_nets), [0]):
                                for pn in pick(pp>1, range(num_nets), [0]):
                                  for dn in pick(dp>1, range(num_nets), [0]):
                                    exe_count += 1
                                    exe_json = {
                                      'num_procs': args.num_procs,
                                      'tensor_par': tp,
                                      'pipeline_par': pp,
                                      'data_par': dp,
                                      'tensor_par_net': tn,
                                      'pipeline_par_net': pn,
                                      'data_par_net': dn,
                                      'batch_size': batch_size,
                                      'microbatch_size': microbatch_size,
                                      'datatype': 'bfloat16',
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

                                    if not args.debug:
                                      try:
                                        logger = logging.Logger('sub')
                                        model = Llm(app, logger)
                                        model.compile(
                                          syst,
                                          Llm.Execution(exe_json))
                                        model.run(syst)
                                        stats = model.get_stats_json()
                                        good_exe_count += 1
                                        if (best_rate == None or
                                            stats['sample_rate'] > best_rate):
                                          best_rate = stats['sample_rate']
                                          best_exe = exe_json
                                          best_stats = stats
                                        if args.csv:
                                          print(exe_count)
                                          write_csv_data(csv, exe_count, exe_json, stats)
                                      except Llm.Error as ex:
                                        logger = logging.getLogger()
                                        logger.debug(f'JSON:{exe_json}\nERROR:{ex}\n')
                                        bad_exe_count += 1
    end_time = datetime.datetime.now()

    if args.csv:
      csv.close()

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

calculon.CommandLine.register(AllExecutions)
