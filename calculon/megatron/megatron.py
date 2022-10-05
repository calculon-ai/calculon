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
    def __init__(self, cfg):
      self.name = cfg['name']
      self.hidden = cfg['hidden']
      self.seq_size = cfg['seq_size']
      self.attn_heads = cfg['attn_heads']
      self.num_layers = cfg['num_layers']

  class Execution:
    """Specifies the execution configuration."""
    def __init__(self, cfg):
      self.num_procs = cfg['num_procs']
      self.tensor_par = cfg['tensor_par']
      self.pipeline_par = cfg['pipeline_par']
      self.data_par = cfg['data_par']
      assert self.num_procs == self.tensor_par * self.pipeline_par * \
        self.data_par, 'tensor * pipeline * data parallelism != num_procs'
      self.batch_size = cfg['batch_size']
      self.minibatch_size = cfg['minibatch_size']
      assert self.batch_size % self.data_par == 0
      assert (self.batch_size // self.data_par) % self.minibatch_size == 0
      self.datatype = cfg['datatype']
      self.activation_recompute = cfg['activation_recompute']
      assert self.activation_recompute in ['full', 'partial', 'none']
      self.pipeline_interleaving = cfg['pipeline_interleaving']
      assert self.pipeline_interleaving > 0, \
        f'Bad pipeline interleaving of {self.pipeline_interleaving}'
      self.optimizer_sharding = cfg['optimizer_sharding']
      self.sequence_par = cfg['sequence_par']
      self.p2p_rs_ag = cfg['p2p_rs_ag']
      self.data_par_overlap = cfg['data_par_overlap']
      self.weight_offload = cfg['weight_offload']
      self.activations_offload = cfg['activations_offload']
      self.optimizer_offload = cfg['optimizer_offload']
      self.training = cfg['training']
      if self.activations_offload:
        assert self.activation_recompute != 'full'

  # This is used for errors where the user may not be fully aware of
  # limitations. Use it like this:
  #   raise self.Error(f'Foo bar {num1} is not {num2}')
  class Error(Exception):
    pass

  @staticmethod
  def _factors(x):
    for cand in range(1, x+1):
      if x % cand == 0:
        yield cand

  @staticmethod
  def get_all_tensor_parallelisms(num_procs):
    yield from Megatron._factors(num_procs)

  @staticmethod
  def get_all_pipeline_parallelisms(num_procs, tensor_par, num_layers):
    assert num_procs % tensor_par == 0
    max_pp = min(num_procs // tensor_par, num_layers)
    yield from Megatron._factors(max_pp)

  @staticmethod
  def get_all_data_parallelisms(num_procs, tensor_par, pipeline_par):
    assert num_procs % (tensor_par * pipeline_par) == 0
    yield num_procs // (tensor_par * pipeline_par)

  @staticmethod
  def get_valid_pipeline_interleavings(num_layers, pipeline_par):
    assert num_layers % pipeline_par == 0
    if pipeline_par == 1:
      yield 1
    else:
      max_ppint = num_layers // pipeline_par
      yield from Megatron._factors(max_ppint)

  @staticmethod
  def get_valid_minibatch_sizes(data_par, batch_size):
    assert batch_size % data_par == 0
    local_batch_size = batch_size // data_par
    yield from Megatron._factors(local_batch_size)

  # TODO refactor to be a member of Application class
  def __init__(self, app, log):
    assert isinstance(app, self.Application)
    self.app = app
    self.log = log

    # Set during compile
    self.exe = None

    # Set during run
    self.sys = None

    # State of calling compile() and run()
    self._compiled = False
    self._executed = False

    # TODO generalize layers to be a graph
    self.megatron_block = []

    # HW parameters to populate during run
    self.vector_throughput = 0
    self.matrix_throughput = 0
    self.mem_throughput = 0
    self.offload_throughput = 0
    self.tp_net_tier = 0
    self.dp_net_tier = 0
    self.pp_net_tier = 0

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
    self.minibatch_recomm_tp_time = 0
    self.minibatch_bw_tp_size = 0
    self.minibatch_bw_tp_time = 0
    self.minibatch_fw_pp_size = 0
    self.minibatch_fw_pp_time = 0
    self.minibatch_bw_pp_size = 0
    self.minibatch_bw_pp_time = 0

    # metrics collected after run for each batch on through a single block
    self.block_weight_space = 0
    self.block_act_space = 0
    self.block_act_checkpoint_size = 0
    self.block_weight_grad_space = 0
    self.block_weight_grad_space_no_sharding = 0
    self.block_act_grad_space = 0
    self.block_optimizer_space = 0
    self.block_fw_flops = 0
    self.block_fw_flops_time = 0
    self.block_fw_mem_accessed = 0
    self.block_fw_mem_time = 0
    self.block_bw_flops = 0
    self.block_bw_flops_time = 0
    self.block_bw_mem_accessed = 0
    self.block_bw_mem_time = 0
    self.block_recompute_time = 0
    self.block_recomm_time = 0
    self.block_tp_comm_size = 0
    self.block_tp_comm_time = 0

    # metrics collected after run for each batch on a single processor
    self.proc_weight_space = 0
    self.proc_act_space = 0
    self.proc_act_checkpoint_size = 0
    self.proc_weight_grad_space = 0
    self.proc_weight_grad_space_no_sharding = 0
    self.proc_act_grad_space = 0
    self.proc_optimizer_space = 0
    self.proc_fw_flops = 0
    self.proc_fw_flops_time = 0
    self.proc_fw_mem_accessed = 0
    self.proc_fw_mem_time = 0
    self.proc_bw_flops = 0
    self.proc_bw_flops_time = 0
    self.proc_bw_mem_accessed = 0
    self.proc_bw_mem_time = 0
    self.proc_recompute_time = 0
    self.proc_recomm_time = 0
    self.proc_bubble_time = 0
    self.proc_tp_comm_size = 0
    self.proc_tp_comm_time = 0
    self.proc_pp_comm_size = 0
    self.proc_pp_comm_time = 0
    self.proc_dp_comm_size = 0
    self.proc_dp_comm_time = 0

  def get_json(self):
    assert self._executed
    j = {}
    j['proc_weight_space'] = self.get_proc_weight_space()
    j['proc_act_space'] = self.get_proc_act_space()
    j['proc_act_checkpoint_size'] = self.get_proc_act_checkpoint_size()
    j['proc_act_grad_space'] = self.get_proc_act_grad_space()
    j['proc_weight_grad_space'] = self.get_proc_weight_grad_space()
    j['proc_optimizer_space'] = self.get_proc_optimizer_space()
    j['proc_fw_time'] = self.get_proc_fw_time()
    j['proc_bw_time'] = self.get_proc_bw_time()
    j['proc_recompute_time'] = self.get_proc_recompute_time()
    j['proc_recomm_time'] = self.get_proc_recomm_time()
    j['proc_bubble_time'] = self.get_proc_bubble_time()
    j['proc_tp_comm_time'] = self.get_proc_tp_comm_time()
    j['proc_pp_comm_time'] = self.get_proc_pp_comm_time()
    j['proc_dp_comm_time'] = self.get_proc_dp_comm_time()
    j['proc_total_time'] = self.get_proc_total_time()
    j['act_offload_bw_req'] = self.get_act_offload_bw_req()
    j['weight_offload_bw_req'] = self.get_weight_offload_bw_req()
    j['optim_offload_bw_req'] = self.get_optim_offload_bw_req()
    j['offload_mem_bw_req'] = self.get_offload_mem_bw_req()
    j['proc_mem_tier1_cap_req'] = self.get_proc_mem_tier1_cap_req()
    j['proc_mem_tier2_cap_req'] = self.get_proc_mem_tier2_cap_req()
    j['useful_flops'] = self.get_useful_flops()
    j['compute_efficiency'] = self.get_compute_efficiency()
    j['system_efficiency'] = self.get_system_efficiency()
    j['total_efficiency'] = self.get_total_efficiency()
    j['layers'] = []
    for layer in self.megatron_block:
      j['layers'].append(layer.get_json())
    return j

  def _build_attn_block(self):
    recompute_flag = self.exe.activation_recompute == "full"
    recompute_attn_flag = self.exe.activation_recompute in ["full", "partial"]

    if self.exe.sequence_par:
      self.megatron_block.append(LayerNorm(
        "AttnBlock_LayerNorm",
        self.seq_par_activation_size,
        self.app.hidden,
        needs_recompute=recompute_flag))
    else:
      self.megatron_block.append(LayerNorm(
        "AttnBlock_LayerNorm",
        self.activation_size,
        self.app.hidden,
        needs_recompute=recompute_flag))
    self.megatron_block.append(TPComm(
      "AttnBlock_F",
      self.activation_size,
      self.exe.tensor_par,
      split_comm=(self.exe.sequence_par or self.exe.p2p_rs_ag),
      conjugate=False,
      needs_recompute=recompute_flag))
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
      conjugate=True,
      needs_recompute=recompute_flag))
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
      self.megatron_block.append(LayerNorm(
        "MlpBlock_LayerNorm",
        self.seq_par_activation_size,
        self.app.hidden,
        needs_recompute=recompute_flag))
    else:
      self.megatron_block.append(LayerNorm(
        "MlpBlock_LayerNorm",
        self.activation_size,
        self.app.hidden,
        needs_recompute=recompute_flag))
    self.megatron_block.append(TPComm(
      "MLPBlock_F",
      self.activation_size,
      self.exe.tensor_par,
      split_comm=(self.exe.sequence_par or self.exe.p2p_rs_ag),
      conjugate=False,
      needs_recompute=recompute_flag))
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
      conjugate=True,
      needs_recompute=recompute_flag))

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
    assert not self._compiled
    assert isinstance(exe, self.Execution)
    self.exe = exe

    self.num_minibatches = self.exe.batch_size / self.exe.data_par / \
      self.exe.minibatch_size
    if self.app.num_layers % self.exe.pipeline_par != 0:
      raise self.Error('Pipeline parallelism must evenly divide the number of '
                       'layers')
    self.layers_per_proc = self.app.num_layers / self.exe.pipeline_par
    if self.exe.pipeline_interleaving > self.layers_per_proc:
      raise self.Error('Pipeline interleaving must be less than or equal to '
                       'the number of layers per processor')
    if self.layers_per_proc % self.exe.pipeline_interleaving != 0:
      raise self.Error('Pipeline interleaving must be a factor value of the '
                       'number of layers per processor')
    self.bytes_per_element = self.types_size_dict[self.exe.datatype]

    # Checks that enough layers per processor exist if offloading is being
    # performed
    if (self.exe.weight_offload or self.exe.activations_offload or
        self.exe.optimizer_offload) and (self.layers_per_proc <= 2):
      raise self.Error('Offloading requires each processor to handle at least 3'
                       ' layers')

    # Build model during the compilation step
    self.batch_seq = self.exe.minibatch_size * self.app.seq_size
    self.activation_size = self.batch_seq * self.app.hidden
    self.batch_seq_par = self.batch_seq / self.exe.tensor_par
    self.seq_par_activation_size = self.batch_seq_par * self.app.hidden
    self._build_attn_block()
    self._build_mlp_block()
    for layer in self.megatron_block:
      layer.set_bytes_per_element(self.bytes_per_element)
      if self.exe.optimizer_sharding:
        layer.shard_optimizer(self.exe.data_par)
    self._compiled = True

  def _update_hw_throughput(self):
    # Determines compute and memory throughputs
    self.vector_throughput = self.sys.compute_throughput('vector')
    self.matrix_throughput = self.sys.compute_throughput('matrix')
    self.mem_throughput = self.sys.memory_throughput(1)
    self.offload_throughput = self.sys.memory_throughput(2)

    # Determines network tier for TP
    net_tier1_size = self.sys.network_size(1)
    net_tier2_size = self.sys.network_size(2)
    assert (self.exe.tensor_par <= net_tier1_size or
            self.exe.tensor_par <= net_tier2_size), \
            f"t={self.exe.tensor_par} is larger than the network " \
            f"size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    if self.exe.tensor_par <= net_tier1_size:
      self.tp_net_tier = 1
    else:
      self.tp_net_tier = 2

    # Determines network tier for DP
    assert (self.exe.data_par * self.exe.tensor_par <= net_tier1_size or
            self.exe.data_par * self.exe.tensor_par <= net_tier2_size), \
            f"d={self.exe.data_par} x t={self.exe.tensor_par} is larger than the " \
            f"network size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    if self.exe.data_par * self.exe.tensor_par <= net_tier1_size:
      self.dp_net_tier = 1
    else:
      self.dp_net_tier = 2

    # Determines network tier for PP
    assert (self.exe.pipeline_par * self.exe.data_par * self.exe.tensor_par <= net_tier1_size or
            self.exe.pipeline_par * self.exe.data_par * self.exe.tensor_par <= net_tier2_size), \
            f"p={self.exe.pipeline_par} x d={self.exe.data_par} x t={self.exe.tensor_par} is larger than the " \
            f"network size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    if (self.exe.pipeline_par * self.exe.data_par * self.exe.tensor_par <=
        net_tier1_size):
      self.pp_net_tier = 1
    else:
      self.pp_net_tier = 2


  def _compute_minibatch_stats(self):
    self.log.debug("%s %s", "vector_throughput:",
      human_format(self.vector_throughput, 'throughput'))
    self.log.debug("%s %s", "matrix_throughput:",
      human_format(self.matrix_throughput, 'throughput'))
    self.log.debug("%s %s", "mem_throughput:",
      human_format(self.mem_throughput, 'bandwidth'))
    self.log.debug("%s %s", "offload_throughput:",
      human_format(self.offload_throughput, 'bandwidth'))

    for layer in self.megatron_block:
      flops_throughput = self.vector_throughput
      if isinstance(layer, Linear):
        flops_throughput = self.matrix_throughput
      # Add flops/bytes/times per layer
      self.minibatch_fw_flops += layer.get_fw_flops()
      self.minibatch_fw_flops_time += \
        layer.get_fw_flops() / flops_throughput
      self.minibatch_fw_mem_accessed += layer.get_fw_mem_accessed()
      self.minibatch_fw_mem_time += \
        layer.get_fw_mem_accessed() / self.mem_throughput
      self.minibatch_bw_flops += layer.get_bw_flops()
      self.minibatch_bw_flops_time += \
        layer.get_bw_flops() / flops_throughput
      self.minibatch_bw_mem_accessed += layer.get_bw_mem_accessed()
      self.minibatch_bw_mem_time += \
        layer.get_bw_mem_accessed() / self.mem_throughput
      self.minibatch_recompute_time += layer.get_recompute_flag() * (
        layer.get_fw_flops() / flops_throughput + \
        layer.get_fw_mem_accessed() / self.mem_throughput)
      self.minibatch_recompute_mem_saving += layer.get_recompute_flag() * (
        layer.get_activation())
      self.block_weight_space += layer.get_weight()
      self.block_act_space += layer.get_activation()
      self.block_weight_grad_space += layer.get_weight_grad()
      self.block_weight_grad_space_no_sharding += layer.get_weight_grad(
        sharded=False)
      self.block_act_grad_space += layer.get_activation_grad()
      self.block_optimizer_space += layer.get_optimizer()

      self.log.debug("%s %s %s", layer.name, 'FW flops:',
        human_format(layer.get_fw_flops(), 'flops'))
      self.log.debug("%s %s %.3e", layer.name, 'FW flops time:',
        layer.get_fw_flops() / flops_throughput)
      self.log.debug("%s %s %s", layer.name, 'FW num inputs:',
        human_format(layer.inputs_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW num output:',
        human_format(layer.output_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW num weights:',
        human_format(layer.weight_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW mem:',
        human_format(layer.get_fw_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %.3e", layer.name, 'FW mem time:',
        layer.get_fw_mem_accessed() / self.mem_throughput)
      self.log.debug("%s %s %.3e", layer.name, 'FW time:',
        layer.get_fw_flops() / flops_throughput + \
        layer.get_fw_mem_accessed() / self.mem_throughput)
      self.log.debug("%s %s %s", layer.name, 'BW flops:',
        human_format(layer.get_bw_flops(), 'flops'))
      self.log.debug("%s %s %.3e", layer.name, 'BW flops time:',
        layer.get_bw_flops() / flops_throughput)
      self.log.debug("%s %s %s", layer.name, 'BW num Wgrads:',
        human_format(layer.weight_grads, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW num Agrads:',
        human_format(layer.activation_grads, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW num Igrads:',
        human_format(layer.output_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW mem:',
        human_format(layer.get_bw_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %.3e", layer.name, 'BW mem time:',
        layer.get_bw_mem_accessed() / self.mem_throughput)
      self.log.debug("%s %s %.3e", layer.name, 'Recompute time:',
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
        human_format(layer.get_optimizer(), 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Weight:',
        human_format(self.block_weight_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act:',
        human_format(self.block_act_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Recompute Saving:',
        human_format(self.minibatch_recompute_mem_saving, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Weight grad:',
        human_format(self.block_weight_grad_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act grad:',
        human_format(self.block_act_grad_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Optim:',
        human_format(self.block_optimizer_space, 'bytes'))

    # TP involves 2 AllReduce (or ReduceScatter + AllGather) collectives for
    # each megatron block, and same amount of communication during the BW pass
    # in case of sequence parallelism, we still communicating
    # full activation sizes, but cannot see the effect from SHARP
    if self.exe.tensor_par > 1:
      if self.exe.sequence_par or self.exe.p2p_rs_ag:
        self.minibatch_fw_tp_size = 2*2 * self.bytes_per_element * \
          self.activation_size
      else:
        self.minibatch_fw_tp_size = 2*2 * self.bytes_per_element * \
          self.activation_size

    if not self.exe.sequence_par and not self.exe.p2p_rs_ag:
      self.minibatch_fw_tp_time = self.sys.network_time(
        self.tp_net_tier, 'all_reduce', self.minibatch_fw_tp_size)
    else:
      self.minibatch_fw_tp_time = \
        self.sys.network_time(self.tp_net_tier, 'reduce_scatter',
                              self.minibatch_fw_tp_size) + \
        self.sys.network_time(self.tp_net_tier, 'all_gather',
                              self.minibatch_fw_tp_size)
    if self.exe.activation_recompute == "full":
      self.minibatch_recomm_tp_time = self.minibatch_fw_tp_time

    if self.exe.training:
      self.minibatch_bw_tp_size = self.minibatch_fw_tp_size
      self.minibatch_bw_tp_time = self.minibatch_fw_tp_time
    # PP communication involves pipline_parallelism_factor of point-to-point
    # instructions between GPUs on neighboring pipeline stages during FW pass,
    # and the same amount oof communication during the BW pass
    self.minibatch_fw_pp_size = self.exe.pipeline_interleaving
    if self.exe.p2p_rs_ag:
      self.minibatch_fw_pp_size *= \
        self.bytes_per_element * self.seq_par_activation_size
    else:
      self.minibatch_fw_pp_size *= \
        self.bytes_per_element * self.activation_size

    self.minibatch_fw_pp_time = self.sys.network_time(
      self.pp_net_tier, 'p2p', self.minibatch_fw_pp_size)

    if self.exe.training:
      self.minibatch_bw_pp_size = self.minibatch_fw_pp_size
      self.minibatch_bw_pp_time = self.minibatch_fw_pp_time

    self.log.debug("%s %s", 'TP comm FW size:',
      human_format(self.minibatch_fw_tp_size, 'bytes'))
    self.log.debug("%s %s", 'TP comm FW time:', self.minibatch_fw_tp_time)
    self.log.debug("%s %s", 'TP comm BW size:',
      human_format(self.minibatch_bw_tp_size, 'bytes'))
    self.log.debug("%s %s", 'TP comm BW time:', self.minibatch_bw_tp_time)
    self.log.debug("%s %s", 'PP comm FW size:',
      human_format(self.minibatch_fw_pp_size, 'bytes'))
    self.log.debug("%s %s", 'PP comm FW time:', self.minibatch_fw_pp_time)
    self.log.debug("%s %s", 'PP comm BW size:',
      human_format(self.minibatch_bw_pp_size, 'bytes'))
    self.log.debug("%s %s", 'PP comm BW time:', self.minibatch_bw_pp_time)

  def _compute_batch_stats(self):
    # compute/memory stats
    self.block_fw_flops = self.num_minibatches * self.minibatch_fw_flops
    self.proc_fw_flops = self.layers_per_proc * self.block_fw_flops
    self.block_fw_flops_time = \
      self.num_minibatches * self.minibatch_fw_flops_time
    self.proc_fw_flops_time = self.layers_per_proc * self.block_fw_flops_time
    self.block_fw_mem_accessed = \
      self.num_minibatches * self.minibatch_fw_mem_accessed
    self.proc_fw_mem_accessed = \
      self.layers_per_proc * self.block_fw_mem_accessed
    self.block_fw_mem_time = self.num_minibatches * self.minibatch_fw_mem_time
    self.proc_fw_mem_time = self.layers_per_proc * self.block_fw_mem_time
    self.block_bw_flops = self.num_minibatches * self.minibatch_bw_flops
    self.proc_bw_flops = self.layers_per_proc * self.block_bw_flops
    self.block_bw_flops_time = \
      self.num_minibatches * self.minibatch_bw_flops_time
    self.proc_bw_flops_time = self.layers_per_proc * self.block_bw_flops_time
    self.block_bw_mem_accessed = \
      self.num_minibatches * self.minibatch_bw_mem_accessed
    self.proc_bw_mem_accessed = \
      self.layers_per_proc * self.block_bw_mem_accessed
    self.block_bw_mem_time = self.num_minibatches * self.minibatch_bw_mem_time
    self.proc_bw_mem_time = self.layers_per_proc * self.block_bw_mem_time
    self.block_recompute_time = \
      self.num_minibatches * self.minibatch_recompute_time
    self.proc_recompute_time = self.layers_per_proc * self.block_recompute_time
    # network stats
    self.block_tp_comm_size = self.num_minibatches * (
      self.minibatch_fw_tp_size + self.minibatch_bw_tp_size)
    self.proc_tp_comm_size = self.layers_per_proc * self.block_tp_comm_size
    self.block_tp_comm_time = self.num_minibatches * (
      self.minibatch_fw_tp_time + self.minibatch_bw_tp_time)
    self.proc_tp_comm_time = self.layers_per_proc * self.block_tp_comm_time
    self.block_recomm_time = self.num_minibatches * \
      self.minibatch_recomm_tp_time
    self.proc_recomm_time = self.layers_per_proc * self.block_recomm_time
    self.proc_pp_comm_size = self.num_minibatches * (
      self.minibatch_fw_pp_size + self.minibatch_bw_pp_size)
    self.proc_pp_comm_time = self.num_minibatches * (
      self.minibatch_fw_pp_time + self.minibatch_bw_pp_time)
    # Bubble forms between i-th minibatch FW and BW passes on the 1st GPU.
    # Wiht no interleaving between layers, it includes
    # L/gpu x minibatch_time x (p-1) x Tcycle, where cycle includes both
    # FW and BW passes, TP and PP communication for FW and BW passes
    # With full interleaving, we only need minibatch_time x (p-1) x Tcycle time
    self.proc_bubble_time = (self.exe.pipeline_par - 1) * (
      self.layers_per_proc / self.exe.pipeline_interleaving * (
        self.minibatch_fw_flops_time + self.minibatch_fw_mem_time +
        self.minibatch_bw_flops_time + self.minibatch_bw_mem_time +
        self.minibatch_recompute_time + self.minibatch_recomm_tp_time +
        self.minibatch_fw_tp_time + self.minibatch_bw_tp_time) +
      self.minibatch_fw_pp_time + self.minibatch_bw_pp_time)

    # Determines how long it takes to perform the DP per layer
    if self.exe.optimizer_sharding:
      # When performing optimizer sharding, the communication time is a
      # reduce-scatter plus an all-gather.
      # TODO(misaev): is the AG the same size as RS?
      self.proc_dp_comm_size = self.block_weight_space * 2
      layer_dp_effective_time = \
        self.sys.network_time(self.dp_net_tier, 'reduce_scatter',
                              self.block_weight_space) + \
        self.sys.network_time(self.dp_net_tier, 'all_gather',
                              self.block_weight_space)
    else:
      # When not performing optimizer sharding, the communication time is a
      # single all-reduce.
      self.proc_dp_comm_size = self.block_weight_space
      layer_dp_effective_time = self.sys.network_time(
        self.dp_net_tier, 'all_reduce', self.proc_dp_comm_size)

    # DP overlap happens if DP time for a previous layer(s) is lower than
    # minibatch BW pass time for next pack of consequtive layers
    # In case of no interleaving, we move a single minibatch through each layer
    # and need to overlap DP during a single layer single minibatch time
    # In case of full interleaving, we propagate p minibatches through each
    # layer and need to overlap DP comm with p-1 minibatches over a layer
    # In a mixed case, we can overlap DP communication of several
    # non-interleaved layers (L/gpu / interleaving_factor) over BW pass of
    # p-1 minibatches through the same amount of layers if memory capacity is
    # enough, or perform offload/prefetch after each layer-minibatch
    # For simplicity we count only bandwidth-optimal case
    if self.exe.data_par_overlap:
      exposed_time = (self.exe.pipeline_interleaving - 1) * max(
        0, layer_dp_effective_time - (self.minibatch_bw_flops_time + \
          self.minibatch_bw_mem_time) * self.exe.pipeline_par * \
          self.layers_per_proc / self.exe.pipeline_interleaving)
      self.proc_dp_comm_time = layer_dp_effective_time + exposed_time
    else:
      self.proc_dp_comm_time = self.layers_per_proc * layer_dp_effective_time

    # memory capacity stats
    self.proc_weight_space = self.block_weight_space * self.layers_per_proc
    # account for activation recomputation
    # for full recompute we keep single layer's activations
    # (no scaling by L/gpu)
    if self.exe.activation_recompute == "full":
      assert self.block_act_space == self.minibatch_recompute_mem_saving, \
        "We expect with full act recomputation we recompute ALL activations"
    else:
      # with partial activation recomputation we need to reclaim memory
      if self.exe.activation_recompute == "partial":
        self.block_act_space -= self.minibatch_recompute_mem_saving
      # Without full recompute, we keep activations for all layers on the GPU
      self.proc_act_space = self.block_act_space * self.layers_per_proc
      # Keep activations for all pipeline stages for PP
      if self.exe.pipeline_interleaving > 1:
        self.proc_act_space *= self.exe.pipeline_par * (
          1 + (self.exe.pipeline_par - 1) / (self.exe.pipeline_interleaving *
                                             self.exe.pipeline_par))
      else: # self.exe.pipeline_interleaving == 1
        assert self.exe.pipeline_interleaving == 1
        self.proc_act_space *= self.exe.pipeline_par
    # Only need activation grads for a single layer, so it stays unchanged
    self.proc_act_grad_space = self.block_act_grad_space
    # Optimizer split  already accounted for during layers compilation
    # We should keep non-sharded weight grad for a current layer for AllReduce
    # and one that we currently compute, so 2x total
    if self.layers_per_proc == 1:
      self.proc_weight_grad_space = self.block_weight_grad_space_no_sharding
    else:
      self.proc_weight_grad_space = \
        2 * self.block_weight_grad_space_no_sharding + \
        self.block_weight_grad_space * (self.layers_per_proc - 2)
    self.proc_optimizer_space = \
      self.block_optimizer_space * self.layers_per_proc

  def _check_mem_caps(self):
    if self.get_proc_mem_tier1_cap_req() > self.sys.mem_tier1_cap:
      raise self.Error(f'Mem tier1 needs {self.get_proc_mem_tier1_cap_req()} '
                       f'but only has {self.sys.mem_tier1_cap}')
    if self.get_proc_mem_tier2_cap_req() > self.sys.mem_tier2_cap:
      raise self.Error(f'Mem tier2 needs {self.get_proc_mem_tier2_cap_req()} '
                       f'but only has {self.sys.mem_tier2_cap}')

  def _misc_sanity_checks(self):
    if self.exe.tensor_par == 1:
      assert self.get_proc_tp_comm_time() == 0
    if self.exe.pipeline_par == 1:
      assert self.get_proc_pp_comm_time() == 0
    if self.exe.data_par == 1:
      assert self.get_proc_dp_comm_time() == 0

  def run(self, sys):
    assert self._compiled, "You must first call self.compile()"
    assert not self._executed
    assert isinstance(sys, System)
    self.sys = sys
    self._update_hw_throughput()
    self._compute_minibatch_stats()
    self._compute_batch_stats()
    self._check_mem_caps()
    self._misc_sanity_checks()
    # TODO def _compute_offload_requirements(self):
    # TODO incorporate 'weight_offload' and 'activations_offload'/'optimizer_offload'
    self._executed = True

  def _get_fw_offload_size(self):
    fw_offload_size = 0
    if self.exe.weight_offload:
      fw_offload_size += self.block_weight_space
    if self.exe.activations_offload:
      fw_offload_size += self.block_act_space
    return fw_offload_size

  def _get_bw_offload_size(self):
    bw_offload_size = 0
    if self.exe.weight_offload:
      bw_offload_size += self.block_weight_space
    if self.exe.optimizer_offload:
      bw_offload_size += \
        self.block_weight_grad_space + self.block_optimizer_space
    return bw_offload_size

  def get_proc_fw_time(self):
    fw_time = self.proc_fw_mem_time
    fw_time += max(self.proc_fw_flops_time,
                   self._get_fw_offload_size() / self.offload_throughput)
    return fw_time

  def get_proc_bw_time(self):
    if self.exe.training:
      bw_time = self.proc_bw_mem_time
      bw_time += max(self.proc_bw_flops_time,
                    self._get_bw_offload_size() / self.offload_throughput)
      return bw_time
    else:
      return 0

  def get_proc_recompute_time(self):
    return self.proc_recompute_time

  def get_proc_recomm_time(self):
    return self.proc_recomm_time

  def get_proc_bubble_time(self):
    return self.proc_bubble_time

  def get_proc_tp_comm_time(self):
    return self.proc_tp_comm_time

  def get_proc_pp_comm_time(self):
    return self.proc_pp_comm_time

  def get_proc_dp_comm_time(self):
    return self.proc_dp_comm_time

  def get_proc_total_time(self):
    time = self.get_proc_fw_time()
    time += self.get_proc_bw_time()
    time += self.get_proc_recompute_time()
    time += self.get_proc_recomm_time()
    time += self.get_proc_bubble_time()
    time += self.get_proc_tp_comm_time()
    time += self.get_proc_pp_comm_time()
    time += self.get_proc_dp_comm_time()
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
    compute_time = self.get_proc_fw_time() + self.get_proc_bw_time()
    perfect_time = self.layers_per_proc * self.num_minibatches * \
      total_flops / self.sys.matrix_flops
    return perfect_time / compute_time

  def get_system_efficiency(self):
    return (self.get_proc_bw_time() + self.get_proc_fw_time()) / \
      self.get_proc_total_time()

  def get_total_efficiency(self):
    total_flops = self.get_useful_flops()
    perfect_time = self.layers_per_proc * self.num_minibatches * \
      total_flops / self.sys.matrix_flops
    return perfect_time / self.get_proc_total_time()

  def get_proc_weight_space(self):
    return self.proc_weight_space

  def get_proc_act_space(self):
    return self.proc_act_space

  def get_proc_act_checkpoint_size(self):
    if self.exe.activation_recompute != 'full':
      return 0
    return self.bytes_per_element * self.block_act_space * \
      self.layers_per_proc

  def get_proc_weight_grad_space(self):
    return self.proc_weight_grad_space

  def get_proc_act_grad_space(self):
    return self.proc_act_grad_space

  def get_proc_optimizer_space(self):
    return self.proc_optimizer_space

  def _get_proc_mem_cap_reqs(self):
    tier1 = 0
    tier2 = 0
    if self.exe.weight_offload:
      tier1 += self.block_weight_space * 2
      tier2 += self.get_proc_weight_space()
    else:
      tier1 += self.get_proc_weight_space()
    if self.exe.activations_offload:
      tier1 += self.block_act_space * 2
      tier2 += self.get_proc_act_space()
    else:
      tier1 += self.get_proc_act_space()
    tier1 += self.get_proc_act_checkpoint_size()
    if self.exe.optimizer_offload:
      tier1 += self.block_weight_grad_space_no_sharding * 2
      tier1 += self.block_optimizer_space * 2
      tier2 += self.get_proc_weight_grad_space() + \
        self.proc_optimizer_space
    else:
      tier1 += self.get_proc_weight_grad_space() + \
        self.get_proc_optimizer_space()
    tier1 += self.get_proc_act_grad_space()
    return tier1, tier2

  def get_proc_mem_tier1_cap_req(self):
    return self._get_proc_mem_cap_reqs()[0]

  def get_proc_mem_tier2_cap_req(self):
    return self._get_proc_mem_cap_reqs()[1]

  def get_act_offload_bw_req(self):
    # We should be able to offload (write) activation during FW pass and
    # prefetch it (read) during BW pass for layer (i-1)
    # After BW pass activations are discarded
    return self.block_act_space / self.block_fw_flops_time

  def get_weight_offload_bw_req(self):
    # We should be able to offload (write) and prefetch (read) weights both
    # during FW and BW passes for layers (i-1) / (i+1).
    # We always keep weights, they cannot be discarded
    return self.block_weight_space / self.block_fw_flops_time

  def get_optim_offload_bw_req(self):
    # We should be able to offload (write) weight grads and optimizer state
    # and prefetch (read) optimizer state during BW passes for layers
    # (i-1) / (i+1).
    return (self.block_weight_grad_space + self.block_optimizer_space) / \
      self.block_bw_flops_time

  def get_offload_mem_bw_req(self):
    req_bw = max(self._get_fw_offload_size() / self.block_fw_flops_time,
                 self._get_bw_offload_size() / self.block_bw_flops_time)
    return req_bw

  def get_total_weight_space(self):
    return self.exe.num_procs * self.get_proc_weight_space()

  def get_total_act_space(self):
    return self.exe.num_procs * self.get_proc_act_space()

  def get_total_act_checkpoint_size(self):
    return self.exe.num_procs * self.get_proc_act_checkpoint_size()

  def get_total_weight_grad_space(self):
    return self.exe.num_procs * self.get_proc_weight_grad_space()

  def get_total_act_grad_space(self):
    return self.exe.num_procs * self.get_proc_act_grad_space()

  def get_total_optimizer_space(self):
    return self.exe.num_procs * self.get_proc_optimizer_space()

  def display_stats(self):
    stats = "" \
      f"Model {self.app.name}: {self.app.num_layers} layers, " \
      f"hidden={self.app.hidden}, num attn heads: {self.app.attn_heads}\n" \
      f"Run on {self.exe.num_procs} processors with TP={self.exe.tensor_par}, PP={self.exe.pipeline_par}, " \
      f"DP={self.exe.data_par}, {self.layers_per_proc} layers per processor\n" \
      f"Execution: {self.exe};\n" \
      f"System: {self.sys};\n" \
      f"Weights: {human_format(self.get_proc_weight_space(), 'bytes')};\n" \
      f"Act: {human_format(self.get_proc_act_space(), 'bytes')};\n" \
      f"Act CP: {human_format(self.get_proc_act_checkpoint_size(), 'bytes')};\n" \
      f"Act grad: {human_format(self.get_proc_act_grad_space(), 'bytes')};\n" \
      f"Weight grad: {human_format(self.get_proc_weight_grad_space(), 'bytes')};\n" \
      f"Optim space: {human_format(self.get_proc_optimizer_space(), 'bytes')};\n" \
      f"Batch FW time: {self.get_proc_fw_time():.2f};\n" \
      f"Batch BW time: {self.get_proc_bw_time():.2f};\n" \
      f"Batch recompute time: {self.get_proc_recompute_time():.2f};\n" \
      f"Batch recomm time: {self.get_proc_recomm_time():.2f};\n" \
      f"Batch bubble time: {self.get_proc_bubble_time():.2f};\n" \
      f"Batch TP comm time: {self.get_proc_tp_comm_time():.2f};\n" \
      f"Batch PP comm time: {self.get_proc_pp_comm_time():.2f};\n" \
      f"Batch DP comm time: {self.get_proc_dp_comm_time():.2f};\n" \
      f"Batch total time: {self.get_proc_total_time():.2f};\n" \
      f"Activation offload required BW: {human_format(self.get_act_offload_bw_req(), 'bandwidth')};\n" \
      f"Weight offload required BW: {human_format(self.get_weight_offload_bw_req(), 'bandwidth')};\n" \
      f"Optimizer offload required BW: {human_format(self.get_optim_offload_bw_req(), 'bandwidth')};\n" \
      f"Total offload required BW: {human_format(self.get_offload_mem_bw_req(), 'bandwidth')};\n" \
      f"Mem tier1 capacity requirement: {human_format(self.get_proc_mem_tier1_cap_req(), 'bytes')};\n" \
      f"Mem tier2 capacity requirement: {human_format(self.get_proc_mem_tier2_cap_req(), 'bytes')};\n" \
      f"Total Flops per processor: {human_format(self.get_useful_flops(), 'flops')};\n" \
      f"Compute efficiency: {self.get_compute_efficiency()*100:.2f}%;\n" \
      f"System efficiency: {self.get_system_efficiency()*100:.2f}%;\n" \
      f"Total efficiency: {self.get_total_efficiency()*100:.2f}%;\n"
    for layer in self.megatron_block:
      layer.display_stats()
    self.log.info(stats)
