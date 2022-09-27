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

from calculon import *
from .layers import *


class Megatron: # stems from class (ParaGraph)
  """
  A Megatron class that implements transformer with tensor, pipeline, and data
  parallelism.
  We should
  1. Initialize the model with certain model parameters
  2. Compile it with certain optimizations and parallelization strategies
  3. Run on particular hardware system
  """

  # TODO move wherever appropriate, e.g. some config class
  types_size_dict = {
    'float8'    : 1,
    'float16'   : 2,
    'float32'   : 4,
    'bfloat16'  : 2
  }

  class Application:
    """Specifies the application configuration."""
    def __init__(self, kvs):
      self.name = kvs['name']
      self.hidden = kvs['hidden']
      self.seq_size = kvs['seq_size']
      self.attn_heads = kvs['attn_heads']  # NEVER USED!!!
      self.num_layers = kvs['num_layers']

  class Execution:
    """Specifies the execution configuration."""
    def __init__(self, kvs):
      self.num_procs = kvs['num_procs']
      self.tensor_par = kvs['tensor_par']
      self.pipeline_par = kvs['pipeline_par']
      self.data_par = kvs['data_par']
      assert self.num_procs == self.tensor_par * self.pipeline_par * \
        self.data_par, "tensor * pipeline * data parallelism != num_procs"
      self.batch_size = kvs['batch_size']
      self.minibatch_size = kvs['minibatch_size']
      self.datatype = kvs['datatype']
      self.activation_recompute = kvs['activation_recompute']
      assert self.activation_recompute in ["full", "partial", "none"]
      self.pipeline_interleaving = kvs['pipeline_interleaving']
      self.optimizer_sharding = kvs['optimizer_sharding']
      self.in_network_allreduce = kvs['in_network_allreduce']
      self.sequence_par = kvs['sequence_par']
      self.p2p_rs_ag = kvs['p2p_rs_ag']
      self.data_par_overlap = kvs['data_par_overlap']
      self.weight_offload = kvs['weight_offload']
      self.activations_offload = kvs['activations_offload']
      self.optimizer_offload = kvs['optimizer_offload']
      self.training = kvs['training']


  # TODO refactor to be a member of Application class
  def __init__(self, app, log):
    assert isinstance(app, self.Application)
    self.app = app
    self.log = log

    # Set during compile
    self.exe = None

    # Set during run
    self.sys = None

    # TODO generalize layers to be a graph
    self.megatron_block = []

    # HW parameters to populate during run
    self.vector_throughput = 0
    self.matrix_throughput = 0
    self.mem_throughput = 0
    self.offload_throughput = 0
    self.tp_net_throughput = 0
    self.dp_net_throughput = 0
    self.pp_net_throughput = 0

    # metrics collected after run for each minibatch
    self.minibatch_fw_flops = 0
    self.minibatch_fw_flops_time = 0
    self.minibatch_fw_mem_accessed = 0
    self.minibatch_fw_mem_time = 0
    self.minibatch_bw_flops = 0
    self.minibatch_bw_flops_time = 0
    self.minibatch_bw_mem_accessed = 0
    self.minibatch_bw_mem_time = 0
    self.minibatch_recompute_mem_saving = 0
    self.minibatch_recompute_time = 0
    self.minibatch_fw_tp_size = 0
    self.minibatch_fw_tp_time = 0
    self.minibatch_bw_tp_size = 0
    self.minibatch_bw_tp_time = 0
    self.minibatch_fw_pp_size = 0
    self.minibatch_fw_pp_time = 0
    self.minibatch_bw_pp_size = 0
    self.minibatch_bw_pp_time = 0

    # metrics collected after run for each batch on a single GPU
    self.gpu_weight_space = 0
    self.gpu_act_space = 0
    self.gpu_act_checkpoint_size = 0
    self.gpu_weight_grad_space = 0
    self.gpu_act_grad_space = 0
    self.gpu_optimizer_space = 0
    self.gpu_fw_flops = 0
    self.gpu_fw_flops_time = 0
    self.gpu_fw_mem_accessed = 0
    self.gpu_fw_mem_time = 0
    self.gpu_bw_flops = 0
    self.gpu_bw_flops_time = 0
    self.gpu_bw_mem_accessed = 0
    self.gpu_bw_mem_time = 0
    self.gpu_recompute_time = 0
    self.gpu_bubble_time = 0
    self.gpu_tp_comm_size = 0
    self.gpu_tp_comm_time = 0
    self.gpu_pp_comm_size = 0
    self.gpu_pp_comm_time = 0
    self.gpu_dp_comm_size = 0
    self.gpu_dp_comm_time = 0

    self.fw_time = 0
    self.bw_time = 0
    self.recompute_time = 0
    self.bubble_time = 0
    self.tp_comm_time = 0
    self.pp_comm_time = 0
    self.dp_comm_time = 0
    self.total_time = 0
    self.gpu_mem_cap_req = 0
    self.offload_mem_bw_req = 0
    self.total_weight_space = 0
    self.total_act_space = 0
    self.total_weight_grad_space = 0
    self.total_act_grad_space = 0
    self.total_optimizer_space = 0

  def _build_attn_block(self):
    recompute_flag = self.exe.activation_recompute == "full"
    recompute_attn_flag = self.exe.activation_recompute in ["full", "partial"]

    if self.exe.sequence_par:
      self.megatron_block.append(LinearNorm(
        "AttnBlock_LinearNorm",
        self.seq_par_activation_size,
        self.app.hidden,
        needs_recompute=recompute_flag))
    else:
      self.megatron_block.append(LinearNorm(
        "AttnBlock_LinearNorm",
        self.activation_size,
        self.app.hidden,
        needs_recompute=recompute_flag))
    self.megatron_block.append(TPComm(
      "AttnBlock_F",
      self.activation_size,
      self.exe.tensor_par,
      split_comm=(self.exe.sequence_par or self.exe.p2p_rs_ag),
      conjugate=False))
    self.megatron_block.append(Fork(
      "AttnBlock_Fork",
      self.activation_size, 3))
    self.megatron_block.append(Linear(
      "AttnBlock_Key",
      self.batch_seq,
      self.app.hidden,
      self.app.hidden / self.exe.tensor_par,
      needs_recompute=recompute_flag))
    self.megatron_block.append(Linear(
      "AttnBlock_Query",
      self.batch_seq,
      self.app.hidden,
      self.app.hidden / self.exe.tensor_par,
      needs_recompute=recompute_flag))
    self.megatron_block.append(Linear(
      "AttnBlock_Value",
      self.batch_seq,
      self.app.hidden,
      self.app.hidden / self.exe.tensor_par,
      needs_recompute=recompute_flag))
    self.megatron_block.append(BatchMatMul(
      "AttnBlock_Multihead_Key_Query",
      self.app.attn_heads / self.exe.tensor_par,
      self.batch_seq,
      self.app.hidden / self.app.attn_heads,
      self.batch_seq,
      needs_recompute=recompute_attn_flag))
    self.megatron_block.append(SoftMax(
      "AttnBlock_Multihead_SoftMax",
      self.app.attn_heads / self.exe.tensor_par * self.batch_seq**2,
      needs_recompute=recompute_attn_flag))
    self.megatron_block.append(DropOut(
      "AttnBlock_Multihead_DropOut",
      self.app.attn_heads / self.exe.tensor_par * self.batch_seq**2,
      needs_recompute=recompute_attn_flag))
    self.megatron_block.append(BatchMatMul(
      "AttnBlock_Multihead_Attn",
      self.app.attn_heads / self.exe.tensor_par,
      self.batch_seq,
      self.batch_seq,
      self.app.hidden / self.app.attn_heads,
      needs_recompute=recompute_attn_flag))
    self.megatron_block.append(Linear(
      "AttnBlock_MLP",
      self.batch_seq,
      self.app.hidden / self.exe.tensor_par,
      self.app.hidden,
      needs_recompute=recompute_flag))
    self.megatron_block.append(TPComm(
      "AttnBlock_G",
      self.activation_size,
      self.exe.tensor_par,
      split_comm=(self.exe.sequence_par or self.exe.p2p_rs_ag),
      conjugate=True))
    if self.exe.sequence_par:
      self.megatron_block.append(DropOut(
        "AttnBlock_DropOut",
        self.seq_par_activation_size,
        needs_recompute=recompute_flag))
      self.megatron_block.append(ElementWise(
        "AttnBlock_Residual",
        self.seq_par_activation_size,
        self.seq_par_activation_size,
        needs_recompute=recompute_flag))
    else:
      self.megatron_block.append(DropOut(
        "AttnBlock_DropOut",
        self.activation_size,
        needs_recompute=recompute_flag))
      self.megatron_block.append(ElementWise(
      "AttnBlock_Residual",
        self.activation_size,
        self.activation_size,
        needs_recompute=recompute_flag))

  def _build_mlp_block(self):
    recompute_flag = self.exe.activation_recompute == "full"

    if self.exe.sequence_par:
      self.megatron_block.append(LinearNorm(
        "MlpBlock_LinearNorm",
        self.seq_par_activation_size,
        self.app.hidden,
        needs_recompute=recompute_flag))
    else:
      self.megatron_block.append(LinearNorm(
        "MlpBlock_LinearNorm",
        self.activation_size,
        self.app.hidden,
        needs_recompute=recompute_flag))
    self.megatron_block.append(TPComm(
      "MLPBlock_F",
      self.activation_size,
      self.exe.tensor_par,
      split_comm=(self.exe.sequence_par or self.exe.p2p_rs_ag),
      conjugate=False))
    self.megatron_block.append(Linear(
      "MlpBlock_MLP1",
      self.batch_seq,
      self.app.hidden,
      self.app.hidden * 4 / self.exe.tensor_par,
      needs_recompute=recompute_flag))
    self.megatron_block.append(GeLU(
      "MlpBlock_GeLU",
      4 * self.activation_size / self.exe.tensor_par,
      needs_recompute=recompute_flag))
    self.megatron_block.append(Linear(
      "MlpBlock_MLP2",
      self.batch_seq,
      self.app.hidden * 4 / self.exe.tensor_par,
      self.app.hidden,
      needs_recompute=recompute_flag))
    self.megatron_block.append(TPComm(
      "MLPBlock_G",
      self.activation_size,
      self.exe.tensor_par,
      split_comm=(self.exe.sequence_par or self.exe.p2p_rs_ag),
      conjugate=True))

    if self.exe.sequence_par:
      self.megatron_block.append(DropOut(
        "MlpBlock_DropOut",
        self.seq_par_activation_size,
        needs_recompute=recompute_flag))
      self.megatron_block.append(ElementWise(
        "MlpBlock_Residual",
        self.seq_par_activation_size,
        self.seq_par_activation_size,
        needs_recompute=recompute_flag))
    else:
      self.megatron_block.append(DropOut(
        "MlpBlock_DropOut",
        self.activation_size,
        needs_recompute=recompute_flag))
      self.megatron_block.append(ElementWise(
        "MlpBlock_Residual",
        self.activation_size,
        self.activation_size,
        needs_recompute=recompute_flag))

  def compile(self, exe):
    assert isinstance(exe, self.Execution)
    self.exe = exe

    self.num_minibatches = self.exe.batch_size / self.exe.data_par / \
      self.exe.minibatch_size
    self.layers_per_proc = self.app.num_layers / self.exe.pipeline_par
    self.bytes_per_element = self.types_size_dict[self.exe.datatype]

    # Build model during the compilation step
    self.batch_seq = self.exe.minibatch_size * self.app.seq_size
    self.activation_size = self.batch_seq * self.app.hidden
    self.batch_seq_par = self.batch_seq / self.exe.tensor_par
    self.seq_par_activation_size = self.batch_seq_par * self.app.hidden
    self._build_attn_block()
    self._build_mlp_block()
    # TODO add f/g functions to properly account for activation space?
    for layer in self.megatron_block:
      layer.set_bytes_per_element(self.bytes_per_element)
      if self.exe.optimizer_sharding:
        layer.shard_optimizer(self.exe.data_par)
    self._compiled = True

  def _update_hw_throughput(self):
    self.vector_throughput = self.sys.vector_tflops * 1e12 * \
      self.sys.vector_flop_eff
    self.matrix_throughput = self.sys.matrix_tflops * 1e12 * \
      self.sys.matrix_flop_eff
    self.mem_throughput = self.sys.mem_tier1_bw * \
      self.sys.mem_tier1_eff * 1000 ** 3
    self.offload_throughput = self.sys.mem_tier2_bw * \
      self.sys.mem_tier2_eff * 1000 ** 3
    assert (self.exe.tensor_par <= self.sys.net_tier1_size or
            self.exe.tensor_par <= self.sys.net_tier2_size), \
            f"t={self.exe.tensor_par} is larger than the network " \
            f"size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    self.tp_net_throughput = self.sys.net_tier2_bw * \
      self.sys.net_tier2_eff * 1000 ** 3
    if self.exe.tensor_par <= self.sys.net_tier1_size:
      self.tp_net_throughput = self.sys.net_tier1_bw * \
        self.sys.net_tier1_eff * 1000 ** 3
    assert (self.exe.data_par * self.exe.tensor_par <= self.sys.net_tier1_size or
            self.exe.data_par * self.exe.tensor_par <= self.sys.net_tier2_size), \
            f"d={self.exe.data_par} x t={self.exe.tensor_par} is larger than the " \
            f"network size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    self.dp_net_throughput = self.sys.net_tier2_bw * \
      self.sys.net_tier2_eff * 1000 ** 3
    if self.exe.data_par * self.exe.tensor_par <= self.sys.net_tier1_size:
      self.dp_net_throughput = self.sys.net_tier1_bw * \
        self.sys.net_tier1_eff * 1000 ** 3
    assert (self.exe.pipeline_par * self.exe.data_par * self.exe.tensor_par <= self.sys.net_tier1_size or
            self.exe.pipeline_par * self.exe.data_par * self.exe.tensor_par <= self.sys.net_tier2_size), \
            f"p={self.exe.pipeline_par} x d={self.exe.data_par} x t={self.exe.tensor_par} is larger than the " \
            f"network size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    self.pp_net_throughput = self.sys.net_tier2_bw * \
      self.sys.net_tier2_eff * 1000 ** 3
    if self.exe.pipeline_par * self.exe.data_par * self.exe.tensor_par < self.sys.net_tier1_size:
      self.pp_net_throughput = self.sys.net_tier1_bw * \
        self.sys.net_tier1_eff * 1000 ** 3

  def _compute_minibatch_stats(self):
    """
    self.log.debug("%s %s", "vector_throughput:",
      human_format(self.vector_throughput, 'throughput'))
    self.log.debug("%s %s", "matrix_throughput:",
      human_format(self.matrix_throughput, 'throughput'))
    self.log.debug("%s %s", "mem_throughput:",
      human_format(self.mem_throughput, 'bandwidth'))
    self.log.debug("%s %s", "offload_throughput:",
      human_format(self.offload_throughput, 'bandwidth'))
    self.log.debug("%s %s", "tp_net_throughput:",
      human_format(self.tp_net_throughput, 'bandwidth'))
    self.log.debug("%s %s", "pp_net_throughput:",
      human_format(self.pp_net_throughput, 'bandwidth'))
    self.log.debug("%s %s", "dp_net_throughput:",
      human_format(self.dp_net_throughput, 'bandwidth'))
    """

    for layer in self.megatron_block:
      flops_throughput = self.vector_throughput
      if isinstance(layer, Linear):
        flops_throughput = self.matrix_throughput
      # Add flops/bytes/times per layer
      self.minibatch_fw_flops += layer.get_fw_flops()
      self.minibatch_fw_flops_time += \
        layer.get_fw_flops() / flops_throughput
      self.minibatch_fw_mem_accessed = layer.get_fw_mem_accessed()
      self.minibatch_fw_mem_time = \
        layer.get_fw_mem_accessed() / self.mem_throughput
      self.minibatch_bw_flops += layer.get_bw_flops()
      self.minibatch_bw_flops_time += \
        layer.get_bw_flops() / flops_throughput
      self.minibatch_bw_mem_accessed = layer.get_bw_mem_accessed()
      self.minibatch_bw_mem_time = \
        layer.get_bw_mem_accessed() / self.mem_throughput
      self.minibatch_recompute_time = layer.get_recompute_flag() * (
        layer.get_fw_flops() + flops_throughput + \
        layer.get_fw_mem_accessed() / self.mem_throughput)
      self.minibatch_recompute_mem_saving += layer.get_recompute_flag() * (
        layer.get_activation())
      self.gpu_weight_space += layer.get_weight()
      self.gpu_act_space += layer.get_activation()
      self.gpu_weight_grad_space += layer.get_weight_grad()
      self.gpu_act_grad_space += layer.get_activation_grad()
      self.gpu_optimizer_space += layer.get_optim()

      self.log.debug("%s %s %s", layer.name, 'FW flops:',
        human_format(layer.get_fw_flops(), 'flops'))
      self.log.debug("%s %s %.3f", layer.name, 'FW flops time:',
        layer.get_fw_flops() / flops_throughput)
      self.log.debug("%s %s %s", layer.name, 'FW num inputs:',
        human_format(layer.inputs_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW num output:',
        human_format(layer.output_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW num weights:',
        human_format(layer.weight_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW mem:',
        human_format(layer.get_fw_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %.3f", layer.name, 'FW mem time:',
        layer.get_fw_mem_accessed() / self.mem_throughput)
      self.log.debug("%s %s %.3f", layer.name, 'FW time:',
        layer.get_fw_flops() / flops_throughput + \
        layer.get_fw_mem_accessed() / self.mem_throughput)
      self.log.debug("%s %s %s", layer.name, 'BW flops:',
        human_format(layer.get_bw_flops(), 'flops'))
      self.log.debug("%s %s %.3f", layer.name, 'BW flops time:',
        layer.get_bw_flops() / flops_throughput)
      self.log.debug("%s %s %s", layer.name, 'BW num Wgrads:',
        human_format(layer.weight_grads, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW num Agrads:',
        human_format(layer.activation_grads, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW num Igrads:',
        human_format(layer.output_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW mem:',
        human_format(layer.get_bw_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %.3f", layer.name, 'BW mem time:',
        layer.get_bw_mem_accessed() / self.mem_throughput)
      self.log.debug("%s %s %.3f", layer.name, 'Recompute time:',
        layer.get_recompute_flag() * (
          layer.get_fw_flops() / flops_throughput + \
          layer.get_fw_mem_accessed() / self.mem_throughput))
      self.log.debug("%s %s %s", layer.name, 'Recompute mem saving:',
        human_format(layer.get_recompute_flag() * \
          layer.get_activation(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Weight:',
        human_format(layer.get_weight(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Act:',
        human_format(layer.get_activation(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Weight grad:',
        human_format(layer.get_weight_grad(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Act grad:',
        human_format(layer.get_activation_grad(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Optim:',
        human_format(layer.get_optim(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Weight:',
        human_format(self.gpu_weight_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act:',
        human_format(self.gpu_act_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Recompute Saving:',
        human_format(self.minibatch_recompute_mem_saving, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Weight grad:',
        human_format(self.gpu_weight_grad_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act grad:',
        human_format(self.gpu_act_grad_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Optim:',
        human_format(self.gpu_optimizer_space, 'bytes'))

    # in case of sequence parallelism, we still communicating
    # full activation sizes, but cannot see the effect from SHARP
    if self.exe.tensor_par > 1:
      if self.exe.sequence_par or self.exe.p2p_rs_ag:
        self.minibatch_fw_tp_size = 2*2 * self.bytes_per_element * \
          self.seq_par_activation_size
      else:
        self.minibatch_fw_tp_size = 2*2 * self.bytes_per_element * \
          self.activation_size
        if self.exe.in_network_allreduce:
          self.minibatch_fw_tp_size /= 2
          # TODO(misaev): you should only adjust the time, not the size
    self.minibatch_fw_tp_time = \
      self.minibatch_fw_tp_size / self.tp_net_throughput
    if self.exe.training:
      self.minibatch_bw_tp_size = self.minibatch_fw_tp_size
      self.minibatch_bw_tp_time = self.minibatch_fw_tp_time
    self.minibatch_fw_pp_size = self.exe.pipeline_interleaving
    if self.exe.p2p_rs_ag:
      self.minibatch_fw_pp_size *= \
        self.bytes_per_element * self.seq_par_activation_size
    else:
      self.minibatch_fw_pp_size *= \
        self.bytes_per_element * self.activation_size
    self.minibatch_fw_pp_time = \
      self.minibatch_fw_pp_size / self.pp_net_throughput
    if self.exe.training:
      self.minibatch_bw_pp_size = self.minibatch_fw_pp_size
      self.minibatch_bw_pp_time = self.minibatch_fw_pp_time

  def _compute_batch_stats(self):
    # compute/memory stats
    self.gpu_fw_flops = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_fw_flops
    self.gpu_fw_flops_time = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_fw_flops_time
    self.gpu_fw_mem_accessed = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_fw_mem_accessed
    self.gpu_fw_mem_time = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_fw_mem_time
    self.gpu_bw_flops = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_bw_flops
    self.gpu_bw_flops_time = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_bw_flops_time
    self.gpu_bw_mem_accessed = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_bw_mem_accessed
    self.gpu_bw_mem_time = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_bw_mem_time
    self.gpu_recompute_time = self.layers_per_proc * self.num_minibatches *\
      self.minibatch_recompute_time
    # network stats
    self.gpu_tp_comm_size = self.layers_per_proc * self.num_minibatches * (
      self.minibatch_fw_tp_size + self.minibatch_bw_tp_size)
    self.gpu_tp_comm_time = self.layers_per_proc * self.num_minibatches * (
      self.minibatch_fw_tp_time + self.minibatch_bw_tp_time)
    self.gpu_pp_comm_size = \
      self.num_minibatches * self.exe.pipeline_interleaving * (
        self.minibatch_fw_pp_size + self.minibatch_bw_pp_size)
    self.gpu_pp_comm_time = self.num_minibatches * (
      self.minibatch_fw_pp_time + self.minibatch_bw_pp_time)
    self.gpu_bubble_time = (self.exe.pipeline_par - 1) * (
      self.layers_per_proc / self.exe.pipeline_interleaving * (
        self.minibatch_fw_flops_time + self.minibatch_fw_mem_time +
        self.minibatch_bw_flops_time + self.minibatch_bw_mem_time +
        self.minibatch_recompute_time +
        self.minibatch_fw_tp_time + self.minibatch_bw_tp_time) +
      self.minibatch_fw_pp_time + self.minibatch_bw_pp_time)
    self.gpu_dp_comm_size = 2 * self.gpu_weight_space
    # TODO(misaev): only the time should be 2x, not the size
    self.gpu_dp_comm_time = self.gpu_dp_comm_size / self.dp_net_throughput
    if self.exe.in_network_allreduce and not self.exe.optimizer_sharding:
      self.gpu_dp_comm_time /= 2
    if self.exe.data_par_overlap:
      exposed_time = (self.exe.pipeline_par - 1) * max(
        0, self.gpu_dp_comm_size / self.layers_per_proc - (
          self.minibatch_bw_flops_time + \
          self.minibatch_bw_mem_time) * self.exe.pipeline_interleaving)
      self.gpu_dp_comm_size = \
        self.gpu_dp_comm_size / self.layers_per_proc + exposed_time
      # TODO(misaev): time not re-adjusted after changing size????
    # memory capacity stats
    self.gpu_weight_space *= self.layers_per_proc
    # account for activation recomputation
    if self.exe.activation_recompute != "full":
      if self.exe.activation_recompute == "partial":
        self.gpu_act_space -= self.minibatch_recompute_mem_saving
      # TODO(misaev): the comments don't match the if/if logic here.

      # for full recompute we keep single layer's activations,
      # otherwise we keep activations for all layers on the GPU
      self.gpu_act_space *= self.layers_per_proc
      # Keep activations for all pipeline stages for PP
      if self.exe.pipeline_interleaving > 1:
        self.gpu_act_space *= self.exe.pipeline_par * (
          1 + (self.exe.pipeline_par - 1) / (self.exe.pipeline_interleaving *
                                             self.exe.pipeline_par))
      else: # self.pipeline_interleaving == 1
        assert self.exe.pipeline_interleaving == 1
        self.gpu_act_space *= self.exe.pipeline_par
    # Only need activation grads for a single layer
    self.gpu_act_grad_space = self.gpu_act_grad_space  #TODO(misaev): assigning to self?
    # Can utilize optimizer split optimization
    self.gpu_weight_grad_space = self.gpu_weight_grad_space * \
      self.layers_per_proc
    self.gpu_optimizer_space = self.gpu_optimizer_space * self.layers_per_proc
    #if self.exe.optimizer_sharding:
    #  self.gpu_weight_grad_space /= self.exe.data_par
    #  self.gpu_optimizer_space /= self.exe.data_par
    #TODO(misaev): is this needed????

  def run(self, sys):
    assert self._compiled, "You should first call self.compile()"
    # TODO - think about how to implement overlap
    assert isinstance(sys, System)
    self.sys = sys
    self._update_hw_throughput()
    self._compute_minibatch_stats()
    self._compute_batch_stats()
    # TODO def _compute_offload_requirements(self):
    # TODO incorporate 'weight_offload' and 'activations_offload'/'optimizer_offload'
    self._executed = True
    # or make a big ass dict, or csv, or pandas?

  def get_fw_time(self):
    return self.gpu_fw_flops_time + self.gpu_fw_mem_time

  def get_bw_time(self):
    if self.exe.training:
      return self.gpu_bw_flops_time + self.gpu_bw_mem_time
    else:
      return 0

  def get_recompute_time(self):
    return self.gpu_recompute_time

  def get_bubble_time(self):
    return self.gpu_bubble_time

  def get_tp_comm_time(self):
    return self.gpu_tp_comm_time

  def get_pp_comm_time(self):
    return self.gpu_pp_comm_time

  def get_dp_comm_time(self):
    return self.gpu_dp_comm_time

  def get_total_time(self):
    time = self.get_fw_time()
    time += self.get_bw_time()
    time += self.get_recompute_time()
    time += self.get_bubble_time()
    time += self.get_tp_comm_time()
    time += self.get_pp_comm_time()
    time += self.get_dp_comm_time()
    return time

  def get_useful_flops(self):
    total_flops = sum(
      [layer.get_fw_flops() for layer in self.megatron_block])
    if self.exe.training:
      total_flops += sum(
        [layer.get_bw_flops() for layer in self.megatron_block])
    return total_flops

  def get_compute_efficiency(self):
    total_flops = self.get_useful_flops()
    compute_time = self.get_fw_time() + self.get_bw_time()
    perfect_time = self.layers_per_proc * self.num_minibatches * \
      total_flops / (self.sys.matrix_tflops * 1e12)
    return perfect_time / compute_time

  def get_system_efficiency(self):
    return (self.get_bw_time() + self.get_fw_time()) / self.get_total_time()

  def get_total_efficiency(self):
    total_flops = self.get_useful_flops()
    perfect_time = self.layers_per_proc * self.num_minibatches * \
      total_flops / (self.sys.matrix_tflops * 1e12)
    return perfect_time / self.get_total_time()

  def get_gpu_weight_space(self):
    return self.gpu_weight_space

  def get_gpu_act_space(self):
    return self.gpu_act_space

  def get_gpu_act_checkpoint_size(self):
    if self.exe.activation_recompute != 'full':
      return 0
    return self.bytes_per_element * self.activation_size * \
      self.layers_per_proc

  def get_gpu_weight_grad_space(self):
    return self.gpu_weight_grad_space

  def get_gpu_act_grad_space(self):
    return self.gpu_act_grad_space

  def get_gpu_optimizer_space(self):
    return self.gpu_optimizer_space

  def get_gpu_mem_requirements(self):
    mem = self.get_gpu_weight_space() + \
      self.get_gpu_act_space() + \
      self.get_gpu_act_checkpoint_size() + \
      self.get_gpu_weight_grad_space() + \
      self.get_gpu_act_grad_space() + \
      self.get_gpu_optimizer_space()
    return mem

  # TODO ===============================================================
  def get_gpu_mem_cap_req(self):
    return self.gpu_mem_cap_req

  def get_offload_mem_bw_req(self):
    return self.offload_mem_bw_req
  # ====================================================================

  def get_total_weight_space(self):
    return self.exe.num_procs * self.get_gpu_weight_space()

  def get_total_act_space(self):
    return self.exe.num_procs * self.get_gpu_act_space()

  def get_total_act_checkpoint_size(self):
    return self.exe.num_procs * self.get_gpu_act_checkpoint_size()

  def get_total_weight_grad_space(self):
    return self.exe.num_procs * self.get_gpu_weight_grad_space()

  def get_total_act_grad_space(self):
    return self.exe.num_procs * self.get_gpu_act_grad_space()

  def get_total_optimizer_space(self):
    return self.exe.num_procs * self.get_gpu_optimizer_space()

  def display_stats(self):
    stats = "" \
      f"Model {self.app.name}: {self.app.num_layers} layers, " \
      f"hidden={self.app.hidden}, num attn heads: {self.app.attn_heads}\n" \
      f"Run on {self.exe.num_procs} processors with TP={self.exe.tensor_par}, PP={self.exe.pipeline_par}, " \
      f"DP={self.exe.data_par}, {self.layers_per_proc} layers per processor\n" \
      f"Execution: {self.exe};\n" \
      f"System: {self.sys};\n" \
      f"Weights: {human_format(self.get_gpu_weight_space(), 'bytes')};\n" \
      f"Act: {human_format(self.get_gpu_act_space(), 'bytes')};\n" \
      f"Act CP: {human_format(self.get_gpu_act_checkpoint_size(), 'bytes')};\n" \
      f"Act grad: {human_format(self.get_gpu_act_grad_space(), 'bytes')};\n" \
      f"Weight grad: {human_format(self.get_gpu_weight_grad_space(), 'bytes')};\n" \
      f"Optim space: {human_format(self.get_gpu_optimizer_space(), 'bytes')};\n" \
      f"Total mem requirements: {human_format(self.get_gpu_mem_requirements(), 'bytes')};\n" \
      f"Batch FW time: {self.get_fw_time():.2f};\n" \
      f"Batch BW time: {self.get_bw_time():.2f};\n" \
      f"Batch recompute time: {self.get_recompute_time():.2f};\n" \
      f"Batch bubble time: {self.get_bubble_time():.2f};\n" \
      f"Batch TP comm time: {self.get_tp_comm_time():.2f};\n" \
      f"Batch PP comm time: {self.get_pp_comm_time():.2f};\n" \
      f"Batch DP comm time: {self.get_dp_comm_time():.2f};\n" \
      f"Batch total time: {self.get_total_time():.2f};\n" \
      f"Total Flops: {human_format(self.get_useful_flops(), 'flops')};\n" \
      f"Compute efficiency: {self.get_compute_efficiency()*100:.2f}%;\n" \
      f"System eficiency: {self.get_system_efficiency()*100:.2f}%;\n" \
      f"Total efficiency: {self.get_total_efficiency()*100:.2f}%;\n"
    self.log.info(stats)
