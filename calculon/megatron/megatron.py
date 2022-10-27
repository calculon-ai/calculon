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

  _types_size_dict = {
    'float8'    : 1,
    'float16'   : 2,
    'float32'   : 4,
    'bfloat16'  : 2
  }

  class Application:
    """Specifies the application configuration."""
    def __init__(self, cfg):
      self.cfg = cfg
      self.name = cfg['name']
      self.hidden = cfg['hidden']
      self.seq_size = cfg['seq_size']
      self.attn_heads = cfg['attn_heads']
      self.num_blocks = cfg['num_blocks']

    def num_parameters(self):
      # https://cs.stanford.edu/~matei/papers/2021/sc_megatron_lm.pdf
      # Equation 2
      p = 1
      p += 13 / (12.0 * self.hidden)
      p += (51200 + self.seq_size) / (12 * self.num_blocks * self.hidden)
      p *= (12 * self.num_blocks * self.hidden**2)
      return p

  class Execution:
    """Specifies the execution configuration."""
    def __init__(self, cfg):
      self.cfg = cfg
      self.training = cfg['training']
      self.num_procs = cfg['num_procs']
      assert self.num_procs > 0
      self.tensor_par = cfg['tensor_par']
      assert self.tensor_par > 0
      self.pipeline_par = cfg['pipeline_par']
      assert self.pipeline_par > 0
      self.data_par = cfg['data_par']
      assert self.data_par > 0
      assert self.num_procs == self.tensor_par * self.pipeline_par * \
        self.data_par, 'tensor * pipeline * data parallelism != num_procs'
      self.global_batch_size = cfg['batch_size']
      assert self.global_batch_size > 0
      self.microbatch_size = cfg['microbatch_size']
      assert self.microbatch_size > 0
      assert self.global_batch_size % self.data_par == 0
      self._local_batch_size = self.global_batch_size // self.data_par
      assert self._local_batch_size % self.microbatch_size == 0
      self._num_microbatches = self._local_batch_size // self.microbatch_size
      self.datatype = cfg['datatype']
      self.activation_recompute = cfg['activation_recompute']
      assert self.activation_recompute in ['full', 'attn_only', 'none']
      if self.activation_recompute in ['full', 'attn_only']:
        assert self.training, "We only perform recompute during training"
      self.pipeline_interleaving = cfg['pipeline_interleaving']
      assert self.pipeline_interleaving > 0, \
        f'Bad pipeline interleaving of {self.pipeline_interleaving}'
      if self.pipeline_par == 1:
        assert self.pipeline_interleaving == 1, \
        f'Bad pipeline interleaving of {self.pipeline_interleaving} with PP=1'
      self.optimizer_sharding = cfg['optimizer_sharding']
      if self.optimizer_sharding:
        assert self.data_par > 1, "We perform optimizer sharding with DP > 1"
      self.tensor_par_comm_type = cfg['tensor_par_comm_type']
      self.in_network_reduction = False
      assert self.tensor_par_comm_type in ['ar', 'p2p_rs_ag', 'rs_ag']
      self._sequence_par = self.tensor_par_comm_type == 'rs_ag'
      self.seq_par_ag_redo = cfg['seq_par_ag_redo']
      if self.seq_par_ag_redo:
        assert self.tensor_par_comm_type == 'rs_ag', "We only redo AG comm"
        assert self._sequence_par, "We only redo AG with sequence parallelism"
        assert self.activation_recompute != 'full', \
          "We assume no extra AG with full recompute"
      self._pipeline_par_rs_ag = \
        self.tensor_par_comm_type in ['p2p_rs_ag', 'rs_ag']
      self.data_par_overlap = cfg['data_par_overlap']
      if self.data_par_overlap:
        assert self.training, "We only perform DP comm overlap during training"
        assert self.data_par > 1, "We perform DP comm overlap with DP > 1"
      self.weight_offload = cfg['weight_offload']
      self.activations_offload = cfg['activations_offload']
      if self.activation_recompute == 'full':
        assert not self.activations_offload
      self.optimizer_offload = cfg['optimizer_offload']
      if self.optimizer_offload:
        assert self.training, \
          "We only perform optimizer offloading during training"
      if self.activations_offload:
        assert self.activation_recompute != 'full'

  # This is used for errors where the user may not be fully aware of
  # limitations. Use it like this:
  #   raise self.Error(f'Foo bar {num1} is not {num2}')
  class Error(Exception):
    pass

  @staticmethod
  def _factors(x):
    for cand in range(1, x + 1):
      if x % cand == 0:
        yield cand

  @staticmethod
  def get_all_tensor_parallelisms(num_procs, attn_heads):
    for cand in Megatron._factors(num_procs):
      if attn_heads % cand == 0:
        yield cand

  @staticmethod
  def get_all_pipeline_parallelisms(num_procs, tensor_par, num_blocks):
    assert num_procs % tensor_par == 0
    max_pp = min(num_procs // tensor_par, num_blocks)
    yield from Megatron._factors(max_pp)

  @staticmethod
  def get_data_parallelism(num_procs, tensor_par, pipeline_par):
    assert num_procs % (tensor_par * pipeline_par) == 0
    return num_procs // (tensor_par * pipeline_par)

  @staticmethod
  def get_valid_pipeline_interleavings(num_blocks, pipeline_par):
    assert num_blocks % pipeline_par == 0
    if pipeline_par == 1:
      yield 1
    else:
      max_ppint = num_blocks // pipeline_par
      yield from Megatron._factors(max_ppint)

  @staticmethod
  def get_valid_microbatch_sizes(data_par, global_batch_size, pipeline_par):
    assert global_batch_size % data_par == 0
    local_batch_size = global_batch_size // data_par
    yield from Megatron._factors(local_batch_size)

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

    # Holds the layers in a single block
    self._megatron_block = []

    # A chunk is a set of blocks for microbatch before passing to the next
    # processor in the pipeline. Each chunk is modeled as a base
    # block that is repeated N-1 times and followed by 1 edge block.
    # Recommunication time is the same in both base and edge blocks.
    self._blocks_per_proc = None
    self._bubble_reduction_blocks = None
    self._blocks_per_chunk = None
    self._chunks_per_proc = None
    self._baseblocks_per_chunk = None
    self._edgeblocks_per_chunk = None

    # Misc compilation values
    self._bytes_per_element = None
    self._batch_seq = None
    self._batch_seq_par = None
    self._activation_size = None
    self._seq_par_activation_size = None

    # Assignments to specific networks
    self._tp_net_tier = None
    self._tp_net = None
    self._dp_net_tier = None
    self._dp_net = None
    self._pp_net_tier = None
    self._pp_net = None

    # metrics collected after run for each microbatch
    self._block_fw_flops = None
    self._block_fw_flops_time = None
    self._block_fw_mem_accessed = None
    self._block_fw_mem_time = None
    self._block_fw_time = None
    self._block_re_flops = None
    self._block_re_flops_time = None
    self._block_re_mem_accessed = None
    self._block_re_mem_time = None
    self._block_re_time = None
    self._block_recompute_mem_saving = None
    self._block_bw_flops = None
    self._block_bw_flops_time = None
    self._block_bw_mem_accessed = None
    self._block_bw_mem_time = None
    self._block_bw_time = None

    self._block_tp_comm_count = None
    self._block_fw_tp_size = None
    self._block_bw_tp_size = None
    self._block_recomm_size = None
    self._block_fw_pp_size = None
    self._block_bw_pp_size = None
    self._block_dp_size = None
    self._baseblock_fw_time = None
    self._edgeblock_fw_time = None
    self._baseblock_bw_time = None
    self._edgeblock_bw_time = None
    self._block_dp_time = None

    self._block_weight_space = None
    self._block_act_space = None
    self._block_act_checkpoint_size = None
    self._block_weight_grad_space = None
    self._block_weight_grad_space_no_sharding = None
    self._block_act_grad_space = None
    self._block_optimizer_space = None

    # Top level memory usage stats
    self._weight_space = None
    self._act_space = None
    self._act_checkpoint_size = None
    self._weight_grad_space = None
    self._act_grad_space = None
    self._optimizer_space = None

    # Top level throughput stats
    self._fw_flops = None
    self._fw_flops_time = None
    self._fw_mem_accessed = None
    self._fw_mem_time = None
    self._fw_time = None
    self._re_flops = None
    self._re_flops_time = None
    self._re_mem_accessed = None
    self._re_mem_time = None
    self._re_time = None
    self._bw_flops = None
    self._bw_flops_time = None
    self._bw_mem_accessed = None
    self._bw_mem_time = None
    self._bw_time = None

    # Top level network stats
    self._tp_comm_time = None
    self._recomm_time = None
    self._pp_comm_time = None
    self._dp_comm_time = None
    self._bubble_time = None

  def get_json(self):
    assert self._executed
    j = {}
    j['tp_net_tier'] = self._tp_net_tier
    j['dp_net_tier'] = self._dp_net_tier
    j['pp_net_tier'] = self._pp_net_tier

    j['block_fw_flops'] = self._block_fw_flops
    j['block_fw_flops_time'] = self._block_fw_flops_time
    j['block_fw_mem_accessed'] = self._block_fw_mem_accessed
    j['block_fw_mem_time'] = self._block_fw_mem_time
    j['block_fw_time'] = self._block_fw_time
    j['block_re_flops'] = self._block_re_flops
    j['block_re_flops_time'] = self._block_re_flops_time
    j['block_re_mem_accessed'] = self._block_re_mem_accessed
    j['block_re_mem_time'] = self._block_re_mem_time
    j['block_re_time'] = self._block_re_time
    j['block_recompute_mem_saving'] = self._block_recompute_mem_saving
    j['block_bw_flops'] = self._block_bw_flops
    j['block_bw_flops_time'] = self._block_bw_flops_time
    j['block_bw_mem_accessed'] = self._block_bw_mem_accessed
    j['block_bw_mem_time'] = self._block_bw_mem_time
    j['block_bw_time'] = self._block_bw_time

    j['block_fw_tp_size'] = self._block_fw_tp_size
    j['block_bw_tp_size'] = self._block_bw_tp_size
    j['block_recomm_size'] = self._block_recomm_size
    j['block_fw_pp_size'] = self._block_fw_pp_size
    j['block_bw_pp_size'] = self._block_bw_pp_size

    j['block_weight_space'] = self._block_weight_space
    j['block_act_space'] = self._block_act_space
    j['block_act_checkpoint_size'] = self._block_act_checkpoint_size
    j['block_weight_grad_space'] = self._block_weight_grad_space
    j['block_weight_grad_space_no_sharding'] = self._block_weight_grad_space_no_sharding
    j['block_act_grad_space'] = self._block_act_grad_space
    j['block_optimizer_space'] = self._block_optimizer_space

    j['weight_space'] = self.get_weight_space()
    j['act_space'] = self.get_act_space()
    j['act_checkpoint_size'] = self.get_act_checkpoint_size()
    j['act_grad_space'] = self.get_act_grad_space()
    j['weight_grad_space'] = self.get_weight_grad_space()
    j['optimizer_space'] = self.get_optimizer_space()

    j['fw_time'] = self.get_fw_time()
    j['bw_time'] = self.get_bw_time()
    j['recompute_time'] = self.get_recompute_time()
    j['recomm_time'] = self.get_recomm_time()
    j['bubble_time'] = self.get_bubble_time()
    j['tp_comm_time'] = self.get_tp_comm_time()
    j['pp_comm_time'] = self.get_pp_comm_time()
    j['dp_comm_time'] = self.get_dp_comm_time()
    j['total_time'] = self.get_total_time()
    j['act_offload_bw_req'] = self.get_act_offload_bw_req()
    j['weight_offload_bw_req'] = self.get_weight_offload_bw_req()
    j['optim_offload_bw_req'] = self.get_optim_offload_bw_req()
    j['offload_mem_bw_req'] = self.get_offload_mem_bw_req()
    j['proc_mem_tier1_cap_req'] = self.get_mem_tier1_cap_req()
    j['proc_mem_tier2_cap_req'] = self.get_mem_tier2_cap_req()
    j['useful_flops'] = self.get_useful_flops()
    j['compute_efficiency'] = self.get_compute_efficiency()
    j['system_efficiency'] = self.get_system_efficiency()
    j['total_efficiency'] = self.get_total_efficiency()
    j['layers'] = []
    for layer in self._megatron_block:
      j['layers'].append(layer.get_json())
    return j

  def _build_attn_block(self):
    recompute_flag = self.exe.activation_recompute == "full"
    recompute_attn_flag = self.exe.activation_recompute in \
      ["full", "attn_only"]
    recompute_ag_flag = recompute_flag or self.exe.seq_par_ag_redo

    assert self.app.hidden % self.exe.tensor_par == 0, (
      f"We should split hidden={self.app.hidden} between"
      f" {self.exe.tensor_par} TP partitions evenly")
    assert self.app.attn_heads % self.exe.tensor_par == 0, (
      f"We should split {self.app.attn_heads} attn_heads between"
      f" {self.exe.tensor_par} TP partitions evenly")
    if self.exe._sequence_par:
      assert self.app.hidden % self.app.attn_heads == 0, (
        f"We should split hidden={self.app.hidden} between"
        f" {self.app.attn_heads} attn_heads evenly")

    self._megatron_block.append(Fork(
      "AttnBlock_Fork",
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      2,
      needs_recompute=recompute_flag,
      # We account this activation when consider Residual layer
      activation_not_stored=True))
    self._megatron_block.append(LayerNorm(
      "AttnBlock_LayerNorm",
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      self.app.hidden,
      needs_recompute=recompute_flag,
      # We account this activation when consider Residual layer
      activation_not_stored=True))
    self._megatron_block.append(TPComm(
      "AttnBlock_F",
      self._activation_size,
      self.exe.tensor_par,
      # We only compute flops/mem analyzing this layers, comm analyzed later
      # This is conservative estimate that does not consider p2p_rs_ag
      # because we don't differentiate between edge and middle blocks here
      split_comm=self.exe._sequence_par,
      conjugate=False,
      in_network_reduction=self.exe.in_network_reduction,
      needs_recompute=recompute_ag_flag))
    self._megatron_block.append(Fork(
      "AttnBlock_Multihead_Fork",
      self._activation_size,
      3,
      needs_recompute=recompute_flag))
    self._megatron_block.append(Linear(
      "AttnBlock_Key",
      self._batch_seq,
      self.app.hidden,
      self.app.hidden // self.exe.tensor_par,
      needs_recompute=recompute_flag,
      activation_not_stored=True))
    self._megatron_block.append(Linear(
      "AttnBlock_Query",
      self._batch_seq,
      self.app.hidden,
      self.app.hidden // self.exe.tensor_par,
      needs_recompute=recompute_flag,
      activation_not_stored=True))
    self._megatron_block.append(Linear(
      "AttnBlock_Value",
      self._batch_seq,
      self.app.hidden,
      self.app.hidden // self.exe.tensor_par,
      needs_recompute=recompute_flag,
      activation_not_stored=True))
    self._megatron_block.append(BatchMatMul(
      "AttnBlock_Multihead_Key_Query",
      self.exe.microbatch_size * self.app.attn_heads // self.exe.tensor_par,
      self.app.seq_size,
      self.app.hidden // self.app.attn_heads,
      self.app.seq_size,
      needs_recompute=recompute_attn_flag))
    self._megatron_block.append(SoftMax(
      "AttnBlock_Multihead_SoftMax",
      self.app.attn_heads // self.exe.tensor_par * \
        self.app.seq_size**2 * self.exe.microbatch_size,
      needs_recompute=recompute_attn_flag))
    self._megatron_block.append(DropOut(
      "AttnBlock_Multihead_DropOut",
      self.app.attn_heads // self.exe.tensor_par * \
        self.app.seq_size**2 * self.exe.microbatch_size,
      needs_recompute=recompute_attn_flag))
    self._megatron_block.append(BatchMatMul(
      "AttnBlock_Multihead_Attn",
      self.exe.microbatch_size * self.app.attn_heads // self.exe.tensor_par,
      self.app.seq_size,
      self.app.seq_size,
      self.app.hidden // self.app.attn_heads,
      needs_recompute=recompute_attn_flag))
    self._megatron_block.append(Linear(
      "AttnBlock_MLP",
      self._batch_seq,
      self.app.hidden // self.exe.tensor_par,
      self.app.hidden,
      needs_recompute=recompute_flag))
    self._megatron_block.append(TPComm(
      "AttnBlock_G",
      self._activation_size,
      self.exe.tensor_par,
      # We only compute flops/mem analyzing this layers, comm analyzed later
      # This is conservative estimate that does not consider p2p_rs_ag
      # because we don't differentiate between edge and middle blocks here
      split_comm=self.exe._sequence_par,
      conjugate=True,
      in_network_reduction=self.exe.in_network_reduction,
      needs_recompute=recompute_flag,
      # We don't store input to it because on BW we do all_gather
      activation_not_stored=self.exe.seq_par_ag_redo))

    self._megatron_block.append(DropOut(
      "AttnBlock_DropOut",
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      needs_recompute=recompute_flag))
    self._megatron_block.append(ElementWise(
      "AttnBlock_Residual",
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      needs_recompute=recompute_flag))

  def _build_mlp_block(self):
    recompute_flag = self.exe.activation_recompute == "full"
    recompute_ag_flag = recompute_flag or self.exe.seq_par_ag_redo

    self._megatron_block.append(Fork(
      "MlpBlock_Fork",
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      2,
      needs_recompute=recompute_flag,
      activation_not_stored=True))
    self._megatron_block.append(LayerNorm(
      "MlpBlock_LayerNorm",
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      self.app.hidden,
      needs_recompute=recompute_flag,
      # We account this activation when consider Residual layer
      activation_not_stored=True))
    self._megatron_block.append(TPComm(
      "MlpBlock_F",
      self._activation_size,
      self.exe.tensor_par,
      # We only compute flops/mem analyzing this layers, comm analyzed later
      # This is conservative estimate that does not consider p2p_rs_ag
      # because we don't differentiate between edge and middle blocks here
      split_comm=self.exe._sequence_par,
      conjugate=False,
      in_network_reduction=self.exe.in_network_reduction,
      needs_recompute=recompute_ag_flag))
    self._megatron_block.append(Linear(
      "MlpBlock_Mlp1",
      self._batch_seq,
      self.app.hidden,
      self.app.hidden * 4 // self.exe.tensor_par,
      needs_recompute=recompute_flag))
    self._megatron_block.append(GeLU(
      "MlpBlock_GeLU",
      4 * self._activation_size // self.exe.tensor_par,
      needs_recompute=recompute_flag))
    self._megatron_block.append(Linear(
      "MlpBlock_Mlp2",
      self._batch_seq,
      self.app.hidden * 4 // self.exe.tensor_par,
      self.app.hidden,
      needs_recompute=recompute_flag))
    self._megatron_block.append(TPComm(
      "MlpBlock_G",
      self._activation_size,
      self.exe.tensor_par,
      # We only compute flops/mem analyzing this layers, comm analyzed later
      # This is conservative estimate that does not consider p2p_rs_ag
      # because we don't differentiate between edge and middle blocks here
      split_comm=self.exe._sequence_par,
      conjugate=True,
      in_network_reduction=self.exe.in_network_reduction,
      needs_recompute=recompute_flag,
      # We don't store input to it because on BW we do all_gather
      activation_not_stored=self.exe.seq_par_ag_redo))

    self._megatron_block.append(DropOut(
      "MlpBlock_DropOut",
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      needs_recompute=recompute_flag))
    self._megatron_block.append(ElementWise(
      "MlpBlock_Residual",
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      pick(self.exe._sequence_par, self._seq_par_activation_size,
           self._activation_size),
      needs_recompute=recompute_flag))

  def compile(self, exe):
    assert not self._compiled
    assert isinstance(exe, self.Execution)
    self.exe = exe

    # If we have number of blocks not divisible by PP, we can allocate the
    # reminder of the blocks on the first num_block % PP Procs and block
    # "bubbles" on the last PP - (num_block % PP) Procs. To reflect that,
    # we round up blocks_per_prock. We report time for Proc0. In that case
    # its bubble time is `PP - (num_block % PP)` blocks shorter
    self._blocks_per_proc = self.app.num_blocks // self.exe.pipeline_par
    if self.app.num_blocks % self.exe.pipeline_par != 0:
      self._blocks_per_proc += 1
      self._bubble_reduction_blocks = self.exe.pipeline_par - (
        self.app.num_blocks % self.exe.pipeline_par)
    else:
      self._bubble_reduction_blocks = 0
    if self.exe.pipeline_interleaving > self._blocks_per_proc:
      raise self.Error('Pipeline interleaving must be less than or equal to '
                       'the number of blocks per processor')
    if self._blocks_per_proc % self.exe.pipeline_interleaving != 0:
      raise self.Error('Pipeline interleaving must be a factor value of the '
                       'number of blocks per processor')
    self._bytes_per_element = self._types_size_dict[self.exe.datatype]

    # Checks that enough blocks per processor exist if offloading is being
    # performed
    if (self.exe.weight_offload or self.exe.activations_offload or
        self.exe.optimizer_offload) and (self._blocks_per_proc <= 2):
      raise self.Error('Offloading requires each processor to handle at least'
                       ' 3 blocks')

    # A chunk is a set of blocks for microbatch before passing to the next
    # processor in the pipeline. Each chunk is modeled as a base
    # block that is repeated N-1 times and followed by 1 edge block.
    # Recommunication time is the same in both base and edge blocks.
    self._blocks_per_chunk = \
      self._blocks_per_proc // self.exe.pipeline_interleaving
    assert self._blocks_per_proc % self._blocks_per_chunk == 0, \
      "PP interleaving should evenly devide {self._blocks_per_proc} blocks"
    self._chunks_per_proc = self._blocks_per_proc // self._blocks_per_chunk
    assert self._chunks_per_proc == self.exe.pipeline_interleaving, \
      "Number of chunks should be equal to pipeline_interleaving"
    self._baseblocks_per_chunk = self._blocks_per_chunk - 1
    self._edgeblocks_per_chunk = 1

    # Build model during the compilation step
    self._batch_seq = self.exe.microbatch_size * self.app.seq_size
    self._activation_size = self._batch_seq * self.app.hidden
    self._batch_seq_par = self._batch_seq // self.exe.tensor_par
    if self.exe._sequence_par or self.exe._pipeline_par_rs_ag:
      assert self._batch_seq % self.exe.tensor_par == 0, (
        f"We should split batch_seq={self._batch_seq} between"
        f" {self.exe.tensor_par} TP partitions evenly")
    self._seq_par_activation_size = self._batch_seq_par * self.app.hidden
    self._build_attn_block()
    self._build_mlp_block()
    for layer in self._megatron_block:
      layer.set_bytes_per_element(self._bytes_per_element)
      if self.exe.optimizer_sharding:
        layer.shard_optimizer(self.exe.data_par)
    self._compiled = True

  def _set_hardware_attributes(self):
    # TODO(nicmcd): there are possible permutations of T,P,D and the
    # corresponding network tiers that we are missing here. Write an algorithm
    # to search them all and choose the best.
    # It might be true that we can't know the best assignment and that we need
    # to move the assignment into the execution configuration and sweep the
    # permutation from the outside.

    # Determines network tier for TP
    net_tier1_size = self.sys.get_network(1).size
    net_tier2_size = self.sys.get_network(2).size
    assert (self.exe.tensor_par <= net_tier1_size or
            self.exe.tensor_par <= net_tier2_size), \
            f"t={self.exe.tensor_par} is larger than the network " \
            f"size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    if self.exe.tensor_par <= net_tier1_size:
      self._tp_net_tier = 1
    else:
      self._tp_net_tier = 2
    self._tp_net = self.sys.get_network(self._tp_net_tier)

    # Determines network tier for DP
    assert (self.exe.data_par * self.exe.tensor_par <= net_tier1_size or
            self.exe.data_par * self.exe.tensor_par <= net_tier2_size), \
            f"d={self.exe.data_par} x t={self.exe.tensor_par} is larger than the " \
            f"network size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    if self.exe.data_par * self.exe.tensor_par <= net_tier1_size:
      self._dp_net_tier = 1
    else:
      self._dp_net_tier = 2
    self._dp_net = self.sys.get_network(self._dp_net_tier)

    # Determines network tier for PP
    assert (self.exe.pipeline_par * self.exe.data_par * \
            self.exe.tensor_par <= net_tier1_size or \
            self.exe.pipeline_par * self.exe.data_par * \
            self.exe.tensor_par <= net_tier2_size), \
            f"p={self.exe.pipeline_par} x d={self.exe.data_par} x t={self.exe.tensor_par} is larger than the " \
            f"network size {self.sys.net_tier1_size} " \
            f"or {self.sys.net_tier2_size}"
    if (self.exe.pipeline_par * self.exe.data_par * self.exe.tensor_par <=
        net_tier1_size):
      self._pp_net_tier = 1
    else:
      self._pp_net_tier = 2
    self._pp_net = self.sys.get_network(self._pp_net_tier)

  def _compute_block_stats(self):
    """
    This function computes the statistics for one microbatch on a single block.
    This only computes flops, flop time, and communication sizes. Since
    tensor and pipeline parallelism cause different communication operations to
    occur at the full batch level, the communication times are computed later.
    """
    if self.exe.training and self.exe.activation_recompute == "full":
      self._block_act_checkpoint_size = \
        self._activation_size * self._bytes_per_element
    else:
      self._block_act_checkpoint_size = 0

    # Initializes values to zero for accumulation in layer loop
    self._block_fw_flops = 0
    self._block_fw_flops_time = 0
    self._block_fw_mem_accessed = 0
    self._block_fw_mem_time = 0
    self._block_fw_time = 0
    self._block_weight_space = 0
    self._block_act_space = 0
    if self.exe.training:
      self._block_re_flops = 0
      self._block_re_flops_time = 0
      self._block_re_mem_accessed = 0
      self._block_re_mem_time = 0
      self._block_re_time = 0
      self._block_recompute_mem_saving = 0
      self._block_bw_flops = 0
      self._block_bw_flops_time = 0
      self._block_bw_mem_accessed = 0
      self._block_bw_mem_time = 0
      self._block_bw_time = 0
      self._block_weight_grad_space = 0
      self._block_weight_grad_space_no_sharding = 0
      self._block_act_grad_space = 0
      self._block_optimizer_space = 0

    prev_layer_recompute = False
    for layer in self._megatron_block:
      # Add flops/bytes/times per layer
      self._block_fw_flops += layer.get_fw_flops()
      self._block_fw_flops_time += self.sys.compute_flops_time(layer, False)
      self._block_fw_mem_accessed += layer.get_fw_mem_accessed()
      self._block_fw_mem_time += self.sys.compute_mem_time(layer, False)
      self._block_fw_time += self.sys.compute_processing_time(layer, False)
      if self.exe.training:
        if layer.get_recompute_flag():
          self._block_re_flops += self._block_fw_flops
          self._block_re_flops_time += self._block_fw_flops_time
          self._block_re_mem_accessed += self._block_fw_mem_accessed
          self._block_re_mem_time += self._block_fw_mem_time
          self._block_re_time += self.sys.compute_processing_time(layer, False)
        if prev_layer_recompute:
          self._block_recompute_mem_saving += layer.get_activation()
        self._block_bw_flops += layer.get_bw_flops()
        self._block_bw_flops_time += self.sys.compute_flops_time(layer, True)
        self._block_bw_mem_accessed += layer.get_bw_mem_accessed()
        self._block_bw_mem_time += self.sys.compute_mem_time(layer, True)
        self._block_bw_time += self.sys.compute_processing_time(layer, True)

      # Accumulate space requirements per block
      self._block_weight_space += layer.get_weight()
      self._block_act_space += layer.get_activation()
      if self.exe.training:
        self._block_weight_grad_space += layer.get_weight_grad()
        self._block_weight_grad_space_no_sharding += layer.get_weight_grad(
          sharded=False)
        self._block_act_grad_space += layer.get_activation_grad()
        self._block_optimizer_space += layer.get_optimizer()

      self.log.debug("%s %s %s", layer.name, 'FW flops:',
                     human_format(layer.get_fw_flops(), 'flops'))
      self.log.debug("%s %s %s", layer.name, 'FW num inputs:',
                     human_format(layer.inputs_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW num output:',
                     human_format(layer.output_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW num weights:',
                     human_format(layer.weight_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'FW mem:',
                     human_format(layer.get_fw_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %.3e", layer.name, 'FW time:',
                     self.sys.compute_processing_time(layer, False))
      self.log.debug("%s %s %s", layer.name, 'BW flops:',
                     human_format(layer.get_bw_flops(), 'flops'))
      self.log.debug("%s %s %s", layer.name, 'BW num Wgrads:',
                     human_format(layer.weight_grads, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW num Agrads:',
                     human_format(layer.activation_grads, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW num Igrads:',
                     human_format(layer.output_size, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'BW mem:',
                     human_format(layer.get_bw_mem_accessed(), 'bytes'))
      self.log.debug("%s %s %.3e", layer.name, 'BW time:',
                     self.sys.compute_processing_time(layer, True))
      self.log.debug("%s %s %.3e", layer.name, 'Recompute:',
                     layer.get_recompute_flag())
      self.log.debug("%s %s %s", layer.name, 'Recompute mem saving:',
                     human_format(prev_layer_recompute * \
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
                     human_format(self._block_weight_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act:',
                     human_format(self._block_act_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Recompute Saving:',
                     human_format(self._block_recompute_mem_saving, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Weight grad:',
                     human_format(self._block_weight_grad_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Act grad:',
                     human_format(self._block_act_grad_space, 'bytes'))
      self.log.debug("%s %s %s", layer.name, 'Incremental Optim:',
                     human_format(self._block_optimizer_space, 'bytes'))
      prev_layer_recompute = layer.get_recompute_flag()

    # Megatron has 2 communication operations per block when using tensor
    # parallelism
    if self.exe.tensor_par > 1:
      self._block_tp_comm_count = 2
    else:
      self._block_tp_comm_count = 0

    # Sets the TP communication operation size.
    # Note: this is for a single operation but there are two F/G comm
    # instructions in each block.
    if self.exe.tensor_par > 1:
      self._block_fw_tp_size = self._activation_size * self._bytes_per_element
    else:
      self._block_fw_tp_size = 0

    # Sets the PP communication operation size
    if self.exe.pipeline_par > 1:
      if self.exe._pipeline_par_rs_ag:
        self._block_fw_pp_size = self._seq_par_activation_size * \
          self._bytes_per_element
      else:
        self._block_fw_pp_size = self._activation_size * \
          self._bytes_per_element
    else:
      self._block_fw_pp_size = 0

    # Sets the recommunication operation size
    if self.exe.training:
      if self.exe.activation_recompute == "full":
        self._block_recomm_size = self._block_fw_tp_size
      elif self.exe.seq_par_ag_redo:
        # only works when recompuet is attn_only or none with seq_par
        self._block_recomm_size = self._block_fw_tp_size
      else:
        self._block_recomm_size = 0
    else:
      self._block_recomm_size = 0

    # When training, BW sizes for TP and PP are same as FW
    if self.exe.training:
      self._block_bw_tp_size = self._block_fw_tp_size
      self._block_bw_pp_size = self._block_fw_pp_size
    else:
      self._block_bw_tp_size = 0
      self._block_bw_pp_size = 0

    self.log.debug("%s %s", 'TP comm FW size:',
                   human_format(self._block_fw_tp_size, 'bytes'))
    self.log.debug("%s %s", 'PP comm FW size:',
                   human_format(self._block_fw_pp_size, 'bytes'))
    self.log.debug("%s %s", 'TP comm BW size:',
                   human_format(self._block_bw_tp_size, 'bytes'))
    self.log.debug("%s %s", 'PP comm BW size:',
                   human_format(self._block_bw_pp_size, 'bytes'))
    self.log.debug("%s %s", 'TP recomm size:',
                   human_format(self._block_recomm_size, 'bytes'))

  def _compute_batch_stats(self):
    """
    This function computes the statistics for a full batch. This uses the per
    microbatch per block statistics from the prior function (see above).
    """
    # Total stats for compute and memory
    mult = self._blocks_per_proc * self.exe._num_microbatches
    self._fw_flops = mult * self._block_fw_flops
    self._fw_flops_time = mult * self._block_fw_flops_time
    self._fw_mem_accessed = mult * self._block_fw_mem_accessed
    self._fw_mem_time = mult * self._block_fw_mem_time
    self._fw_time = mult * self._block_fw_time
    self._re_flops = mult * self._block_re_flops
    self._re_flops_time = mult * self._block_re_flops_time
    self._re_mem_accessed = mult * self._block_re_mem_accessed
    self._re_mem_time = mult * self._block_re_mem_time
    self._re_time = mult * self._block_re_time
    self._bw_flops = mult * self._block_bw_flops
    self._bw_flops_time = mult * self._block_bw_flops_time
    self._bw_mem_accessed = mult * self._block_bw_mem_accessed
    self._bw_mem_time = mult * self._block_bw_mem_time
    self._bw_time = mult * self._block_bw_time

    # Recommunication is caused by activation recomputation in
    # "full" mode. It is always 2 AllReduce operations
    # We consider full activation recomputation to start with the full
    # activation checkpoint  (not distributed across TP nodes)
    # TODO(misaev): think if we want to introduce act_cp_sharding optimization
    if self.exe.training and self.exe.tensor_par > 1:
      if self.exe.activation_recompute == 'full':
        if self.exe._sequence_par:
          block_recomm_time = self._block_tp_comm_count * (
            self._tp_net.time(
              'reduce_scatter', self._block_recomm_size, self.exe.tensor_par) +
            self._tp_net.time(
              'all_gather', self._block_recomm_size, self.exe.tensor_par))
        else:
          block_recomm_time = self._block_tp_comm_count * self._tp_net.time(
            'all_reduce', self._block_recomm_size, self.exe.tensor_par)
      elif self.exe.seq_par_ag_redo:
        block_recomm_time = self._block_tp_comm_count * self._tp_net.time(
          'all_gather', self._block_recomm_size, self.exe.tensor_par)
      else:
        block_recomm_time = 0
    else:
      block_recomm_time = 0

    # Computes TP communication times base and edge blocks, split into FW and BW
    if self.exe.tensor_par > 1:
      if self.exe._sequence_par:
        baseblock_fw_tp_time = self._block_tp_comm_count * (
          self._tp_net.time(
            'reduce_scatter', self._block_fw_tp_size, self.exe.tensor_par) +
          self._tp_net.time(
            'all_gather', self._block_fw_tp_size, self.exe.tensor_par))
      else:
        baseblock_fw_tp_time = self._block_tp_comm_count * (
          self._tp_net.time(
            'all_reduce', self._block_fw_tp_size, self.exe.tensor_par))
      if self.exe._pipeline_par_rs_ag:
        edgeblock_fw_tp_time = self._block_tp_comm_count * (
          self._tp_net.time(
            'reduce_scatter', self._block_fw_tp_size, self.exe.tensor_par) +
          self._tp_net.time(
            'all_gather', self._block_fw_tp_size, self.exe.tensor_par))
      else:
        edgeblock_fw_tp_time = self._block_tp_comm_count * (
          self._tp_net.time(
            'all_reduce', self._block_fw_tp_size, self.exe.tensor_par))
      if self.exe.training:
        if self.exe._sequence_par:
          baseblock_bw_tp_time = self._block_tp_comm_count * (
            self._tp_net.time(
              'reduce_scatter', self._block_bw_tp_size, self.exe.tensor_par) +
            self._tp_net.time(
              'all_gather', self._block_bw_tp_size, self.exe.tensor_par))
        else:
          baseblock_bw_tp_time = self._block_tp_comm_count * (
            self._tp_net.time(
              'all_reduce', self._block_bw_tp_size, self.exe.tensor_par))
        if self.exe._pipeline_par_rs_ag:
          edgeblock_bw_tp_time = self._block_tp_comm_count * (
            self._tp_net.time(
              'reduce_scatter', self._block_bw_tp_size, self.exe.tensor_par) +
            self._tp_net.time(
              'all_gather', self._block_bw_tp_size, self.exe.tensor_par))
        else:
          edgeblock_bw_tp_time = self._block_tp_comm_count * (
            self._tp_net.time(
              'all_reduce', self._block_bw_tp_size, self.exe.tensor_par))
          assert edgeblock_bw_tp_time == edgeblock_fw_tp_time and \
            baseblock_bw_tp_time == baseblock_fw_tp_time, \
          "We expect TP communication time is the same during FW and BW passes"
      else:
        baseblock_bw_tp_time = 0
        edgeblock_bw_tp_time = 0
    else:
      baseblock_fw_tp_time = 0
      edgeblock_fw_tp_time = 0
      baseblock_bw_tp_time = 0
      edgeblock_bw_tp_time = 0

    # These TP numbers are for total times for all blocks in all chunks
    tp_fw_comm_time = self.exe._num_microbatches * self._chunks_per_proc * (
      (self._baseblocks_per_chunk * baseblock_fw_tp_time) +
      (self._edgeblocks_per_chunk * edgeblock_fw_tp_time))
    tp_bw_comm_time = self.exe._num_microbatches * self._chunks_per_proc * (
      (self._baseblocks_per_chunk * baseblock_bw_tp_time) +
      (self._edgeblocks_per_chunk * edgeblock_bw_tp_time))
    tp_recomm_time = self.exe._num_microbatches * self._chunks_per_proc * (
      self._blocks_per_chunk * block_recomm_time)

    # Per chunk PP comm time
    chunk_fw_pp_time = self._pp_net.time('p2p', self._block_fw_pp_size, 2)
    chunk_bw_pp_time = self._pp_net.time('p2p', self._block_bw_pp_size, 2)

    # Determines number of times PP causes pipeline p2p communications per
    # chunk during the forward and backward pass (equal to chunks per proc)
    if self.exe.pipeline_par > 1:
      num_fw_pp_p2ps = self._chunks_per_proc
      if self.exe.training:
        num_bw_pp_p2ps = self._chunks_per_proc
      else:
        num_bw_pp_p2ps = 0
    else:
      num_fw_pp_p2ps = 0
      num_bw_pp_p2ps = 0

    # These PP numbers are for total times for all blocks and all microbatches
    pp_fw_comm_time = self.exe._num_microbatches * num_fw_pp_p2ps * chunk_fw_pp_time
    pp_bw_comm_time = self.exe._num_microbatches * num_bw_pp_p2ps * chunk_bw_pp_time

    # Aggregrates metrics
    self._tp_comm_time = tp_fw_comm_time + tp_bw_comm_time
    self._recomm_time = tp_recomm_time
    self._pp_comm_time = pp_fw_comm_time + pp_bw_comm_time

    self.log.debug("%s %s", 'TP comm baseblock FW time:', baseblock_fw_tp_time)
    self.log.debug("%s %s", 'TP comm edgeblock FW time:', edgeblock_fw_tp_time)
    self.log.debug("%s %s", 'TP comm FW time:', tp_fw_comm_time)
    self.log.debug("%s %s", 'TP comm baseblock BW time:', baseblock_bw_tp_time)
    self.log.debug("%s %s", 'TP comm edgeblock BW time:', edgeblock_bw_tp_time)
    self.log.debug("%s %s", 'PP comm chunk FW time:', chunk_fw_pp_time)
    self.log.debug("%s %s", 'PP comm chunk BW time:', chunk_bw_pp_time)
    self.log.debug("%s %s", 'TP comm BW time:', tp_bw_comm_time)
    self.log.debug("%s %s", 'PP comm FW time:', pp_fw_comm_time)
    self.log.debug("%s %s", 'PP comm BW time:', pp_bw_comm_time)

    # Bubble forms between i-th microbatch FW and BW passes on the 1st GPU.
    # With no interleaving between blocks, it includes
    # L/gpu x microbatch_time x (p-1) x Tcycle, where cycle includes both
    # FW and BW passes, TP and PP communication for FW and BW passes
    # With full interleaving, we only need microbatch_time x (p-1) x Tcycle time
    self._baseblock_fw_time = self._block_fw_time + baseblock_fw_tp_time
    self._edgeblock_fw_time = (self._block_fw_time + edgeblock_fw_tp_time +
                               chunk_fw_pp_time)
    self._baseblock_bw_time = (self._block_re_time + block_recomm_time +
                               self._block_bw_time + baseblock_bw_tp_time)
    self._edgeblock_bw_time = (self._block_re_time + block_recomm_time +
                               self._block_bw_time + edgeblock_bw_tp_time +
                               chunk_bw_pp_time)
    chunk_fw_time = (
      (self._baseblocks_per_chunk * self._baseblock_fw_time) +
      (self._edgeblocks_per_chunk * self._edgeblock_fw_time))
    chunk_bw_time = (
      (self._baseblocks_per_chunk * self._baseblock_bw_time) +
      (self._edgeblocks_per_chunk * self._edgeblock_bw_time))
    chunk_time = chunk_fw_time + chunk_bw_time
    # Block bubbles appear due to uneven devision of blocks by pipeline stages
    # and result in the schedule bubble shorten by the missing edge blocks on
    # the later pipeline stages
    bubble_reduction_time = self._bubble_reduction_blocks * (
      self._baseblock_bw_time + self._edgeblock_bw_time) / 2
    chunks_in_bubble = self.exe.pipeline_par - 1
    # With PP interleaving we assume that we move through every chunk at least
    # PP mini batches. If num_microbatches < PP, then we have extra bubbles
    if self.exe._num_microbatches < self.exe.pipeline_par:
      extra_interleaving_bubbles = chunks_in_bubble * (
        self.exe.pipeline_par - self.exe._num_microbatches)
    else:
      extra_interleaving_bubbles = 0
    self._bubble_time = chunks_in_bubble * chunk_time - \
      bubble_reduction_time + extra_interleaving_bubbles * chunk_time

    self.log.debug("%s %s", 'Block FW time:', self._block_fw_time)
    self.log.debug("%s %s", 'Baseblock FW time:', self._baseblock_fw_time)
    self.log.debug("%s %s", 'Edgeblock FW time:', self._edgeblock_fw_time)
    self.log.debug("%s %s", 'Block REcomm time:', block_recomm_time)
    self.log.debug("%s %s", 'Block RE time:', self._block_re_time)
    self.log.debug("%s %s", 'Block BW time:', self._block_bw_time)
    self.log.debug("%s %s", 'Baseblock BW time:', self._baseblock_bw_time)
    self.log.debug("%s %s", 'Edgeblock BW time:', self._edgeblock_bw_time)

    # Determines how long it takes to perform the DP per block
    # This assumes no DP communication overlap (will be adjusted later).
    if self.exe.data_par > 1 and self.exe.training:
      self._block_dp_size = self._block_weight_space
      if self.exe.optimizer_sharding:
        # When performing optimizer sharding, the communication time is a
        # reduce-scatter plus an all-gather.
        self._block_dp_time = (
          self._dp_net.time(
            'reduce_scatter', self._block_dp_size, self.exe.data_par) +
          self._dp_net.time(
            'all_gather', self._block_dp_size, self.exe.data_par))
      else:
        # When not performing optimizer sharding, the communication time is a
        # single all-reduce.
        self._block_dp_time = self._dp_net.time(
          'all_reduce', self._block_dp_size, self.exe.data_par)
    else:
      self._block_dp_time = 0
    self.log.debug('DP block comm size: %s',
                   human_format(self._block_dp_size, 'bytes'))
    self.log.debug('DP block comm time (no overlap): %.3e',
                   self._block_dp_time)

    # DP overlap happens if DP time for a previous block(s) is lower than
    # microbatch BW pass time for next pack of consecutive blocks
    # In case of no interleaving, we move a single microbatch through each block
    # and need to overlap DP during a single block single microbatch time
    # In case of full interleaving, we propagate p microbatches through each
    # block and need to overlap DP comm with p-1 microbatches over a block
    # In a mixed case, we can overlap DP communication of several
    # non-interleaved blocks (L/gpu / interleaving_factor) over BW pass of
    # p-1 microbatches through the same amount of blocks if memory capacity is
    # enough, or perform offload/prefetch after each block-microbatch
    # For simplicity we count only bandwidth-optimal case
    if self.exe.data_par > 1 and self.exe.training:
      if self.exe.data_par_overlap:
        # we can evenly overlap all the chunks except for the last one
        # in the ast chunk we can overlap only all blocks except for the last
        num_overlappable_chunks = self.exe.pipeline_interleaving - 1
        last_chunk_overlap_size = self._blocks_per_chunk - 1
        # Overlappable chunks have overlap size equal to
        # blocks_per_chunk * num_microbatches
        # In case of 1F1B schedule, num_microbatches == pipeline_par
        overlappable_chunks_exposed_time = num_overlappable_chunks * \
          max(0, self._blocks_per_chunk * self._block_dp_time - \
            self.exe.pipeline_par * chunk_bw_time)
        # in the last chunk, we overlap DP comm over first edge block and all
        # middle blocks, so we substract the time of the last edge block
        last_chunk_bw_time = chunk_bw_time - chunk_bw_pp_time - (
          self._baseblock_bw_time + self._edgeblock_bw_time) / 2
        last_chunk_exposed_time = max(0, (
          last_chunk_overlap_size * self._block_dp_time - last_chunk_bw_time))
        exposed_time = \
          overlappable_chunks_exposed_time + last_chunk_exposed_time
        self._dp_comm_time = self._block_dp_time + exposed_time
      else:
        self._dp_comm_time = self._blocks_per_proc * self._block_dp_time
    else:
      self._dp_comm_time = 0
    self.log.debug('DP comm time exposed: %.3e', self._dp_comm_time)
    self.log.debug('DP comm time on the link: %.3e',
                   self._blocks_per_proc * self._block_dp_time)

    # memory capacity stats
    self._weight_space = self._block_weight_space * self._blocks_per_proc
    # account for activation recomputation
    # for full recompute we keep single block's activations
    # (no scaling by L/gpu)
    if self.exe.training:
      if self.exe.activation_recompute == "full":
        assert self._block_act_space == self._block_recompute_mem_saving, \
          "We expect with full act recomputation we recompute ALL activations"
        self._act_space = self._block_act_space
        # We need to store checkpoints for all microbatches before we compute
        # BW pass, with 1F1B schedule it is pipeline_par microbatches
        self._act_checkpoint_size = self._blocks_per_proc * \
          self._block_act_checkpoint_size
        # Keep activations for all pipeline stages for PP
        if self.exe.pipeline_interleaving > 1:
          self._act_checkpoint_size *= self.exe.pipeline_par * (
            1 + (self.exe.pipeline_par - 1) / (self.exe.pipeline_interleaving *
                                               self.exe.pipeline_par))
        else:
          assert self.exe.pipeline_interleaving == 1
          self._act_checkpoint_size *= self.exe.pipeline_par
      else:
        # with partial activation recomputation we need to reclaim memory
        if self.exe.activation_recompute == "attn_only":
          self._block_act_space -= self._block_recompute_mem_saving
        # Without full recompute, we keep activations for all blocks on the GPU
        self._act_space = self._block_act_space * self._blocks_per_proc
        # Without full recompute, we don't need checkpoints
        self._act_checkpoint_size = 0
        # Keep activations for all pipeline stages for PP
        if self.exe.pipeline_interleaving > 1:
          self._act_space *= self.exe.pipeline_par * (
            1 + (self.exe.pipeline_par - 1) / (self.exe.pipeline_interleaving *
                                               self.exe.pipeline_par))
        else:
          assert self.exe.pipeline_interleaving == 1
          self._act_space *= self.exe.pipeline_par
      # Only need activation grads for a single block
      self._act_grad_space = self._block_act_grad_space
    else:
      self._act_space = self._block_act_space
      self._act_checkpoint_size = 0
      self._act_grad_space = 0

    # Optimizer split  already accounted for during block compilation
    # We should keep non-sharded weight grad for a current block for AllReduce
    # and one that we currently compute, so 2x total
    # We only need a single no sharded weight grad copy for before reduction
    if self.exe.training:
      if self._blocks_per_proc == 1:
        self._weight_grad_space = self._block_weight_grad_space_no_sharding
      else:
        self._weight_grad_space = \
          self._block_weight_grad_space_no_sharding + \
          self._block_weight_grad_space * (self._blocks_per_proc - 1)
      self._optimizer_space = \
        self._block_optimizer_space * self._blocks_per_proc
    else:
      self._weight_grad_space = 0
      self._optimizer_space = 0

  def _check_mem_caps(self):
    if self.get_mem_tier1_cap_req() > self.sys.mem1.capacity:
      raise self.Error(f'Mem tier1 needs '
                       f'{human_format(self.get_mem_tier1_cap_req(), "bytes")} '
                       f'but only has '
                       f'{human_format(self.sys.mem1.capacity, "bytes")}')
    if self.get_mem_tier2_cap_req() > self.sys.mem2.capacity:
      raise self.Error(f'Mem tier2 needs '
                       f'{human_format(self.get_mem_tier2_cap_req(), "bytes")} '
                       f'but only has '
                       f'{human_format(self.sys.mem2.capacity, "bytes")}')

  def _misc_sanity_checks(self):
    if self.exe.tensor_par == 1:
      assert self.get_tp_comm_time() == 0
    if self.exe.pipeline_par == 1:
      assert self.get_pp_comm_time() == 0
    if self.exe.data_par == 1:
      assert self.get_dp_comm_time() == 0

    assert self._fw_flops >= self._block_fw_flops
    assert self._fw_flops_time >= self._block_fw_flops_time
    assert self._fw_mem_accessed >= self._block_fw_mem_accessed
    assert self._fw_mem_time >= self._block_fw_mem_time
    assert self._fw_time >= self._block_fw_time
    assert self._re_flops >= self._block_re_flops
    assert self._re_flops_time >= self._block_re_flops_time
    assert self._re_mem_accessed >= self._block_re_mem_accessed
    assert self._re_mem_time >= self._block_re_mem_time
    assert self._re_time >= self._block_re_time
    assert self._bw_flops >= self._block_bw_flops
    assert self._bw_flops_time >= self._block_bw_flops_time
    assert self._bw_mem_accessed >= self._block_bw_mem_accessed
    assert self._bw_mem_time >= self._block_bw_mem_time
    assert self._bw_time >= self._block_bw_time
    assert self._weight_space >= self._block_weight_space
    assert self._act_space >= self._block_act_space
    assert self._act_checkpoint_size >= self._block_act_checkpoint_size
    assert self._weight_grad_space >= self._block_weight_grad_space_no_sharding
    assert self._act_grad_space == self._block_act_grad_space
    assert self._optimizer_space >= self._block_optimizer_space

    if not self.exe.training:
      # when not training (inference), backward is not performed and DP has no
      # communication overhead
      assert self.get_bw_time() == 0
      assert self.get_bw_offload_time() == 0
      assert self.get_recompute_time() == 0
      assert self.get_act_checkpoint_size() == 0
      assert self.get_dp_comm_time() == 0
    else:
      # when training, backward is performed
      assert self.get_bw_time() > 0
      if self.exe.activation_recompute == 'full':
        assert self.get_recompute_time() > 0
        assert self.get_act_checkpoint_size() > 0
      elif self.exe.activation_recompute == 'attn_only':
        assert self.get_recompute_time() > 0
        assert self.get_act_checkpoint_size() == 0
      else:
        if not self.exe.seq_par_ag_redo:
          assert self.get_recompute_time() == 0
        assert self.get_act_checkpoint_size() == 0


  def run(self, sys):
    assert self._compiled, "You must first call self.compile()"
    assert not self._executed
    assert isinstance(sys, System)
    self.sys = sys
    self._set_hardware_attributes()
    self._compute_block_stats()
    self._compute_batch_stats()
    self._check_mem_caps()
    self._misc_sanity_checks()
    self._executed = True

  def _get_fw_offload_size(self):
    fw_offload_size = 0
    if self.exe.weight_offload:
      fw_offload_size += self._block_weight_space
    if self.exe.activations_offload:
      fw_offload_size += self._block_act_space
    return fw_offload_size

  def _get_bw_offload_size(self):
    bw_offload_size = 0
    if self.exe.training:
      if self.exe.weight_offload:
        bw_offload_size += self._block_weight_space
      if self.exe.optimizer_offload:
        bw_offload_size += \
          self._block_weight_grad_space + self._block_optimizer_space
    return bw_offload_size

  def get_fw_time(self):
    return self._fw_time

  def get_fw_offload_time(self):
    return self.sys.compute_offload_time(self._get_fw_offload_size())

  def get_fw_offload_overhead(self):
    baseblock_overhead = max(
      0, self.get_fw_offload_time() - self._baseblock_fw_time)
    edgeblock_overhead = max(
      0, self.get_fw_offload_time() - self._edgeblock_fw_time)
    full_overhead = self.exe._num_microbatches * self._chunks_per_proc * (
      (self._baseblocks_per_chunk * baseblock_overhead) +
      (self._edgeblocks_per_chunk * edgeblock_overhead))
    return full_overhead

  def get_bw_time(self):
    return self._bw_time

  def get_bw_offload_time(self):
    if self.exe.training:
      return self.sys.compute_offload_time(self._get_bw_offload_size())
    else:
      return 0

  def get_bw_offload_overhead(self):
    if self.exe.training:
      baseblock_overhead = max(
        0, self.get_bw_offload_time() - self._baseblock_bw_time)
      edgeblock_overhead = max(
        0, self.get_bw_offload_time() - self._edgeblock_bw_time)
      full_overhead = self.exe._num_microbatches * self._chunks_per_proc * (
        (self._baseblocks_per_chunk * baseblock_overhead) +
        (self._edgeblocks_per_chunk * edgeblock_overhead))
      return full_overhead
    else:
      return 0

  def get_recompute_time(self):
    return self._re_time

  def get_recomm_time(self):
    if self.exe.training:
      return self._recomm_time
    else:
      return 0

  def get_bubble_time(self):
    return self._bubble_time

  def get_tp_comm_time(self):
    return self._tp_comm_time

  def get_pp_comm_time(self):
    return self._pp_comm_time

  def get_dp_comm_time(self):
    if self.exe.training:
      return self._dp_comm_time
    else:
      return 0

  def get_dp_comm_net_time(self):
    if self.exe.training:
      return self._blocks_per_proc * self._block_dp_time
    else:
      return 0

  def get_total_time(self):
    time = self.get_fw_time()
    time += self.get_bw_time()
    time += self.get_fw_offload_overhead()
    time += self.get_bw_offload_overhead()
    time += self.get_recompute_time()
    time += self.get_recomm_time()
    time += self.get_bubble_time()
    time += self.get_tp_comm_time()
    time += self.get_pp_comm_time()
    time += self.get_dp_comm_time()
    return time

  def get_useful_flops(self):
    total_flops = sum(
      [block.get_fw_flops() for block in self._megatron_block])
    if self.exe.training:
      total_flops += sum(
        [block.get_bw_flops() for block in self._megatron_block])
    return total_flops

  def get_compute_efficiency(self):
    total_flops = self.get_useful_flops()
    compute_time = self.get_fw_time() + self.get_bw_time()
    perfect_time = self._blocks_per_proc * self.exe._num_microbatches * \
      total_flops / self.sys.matrix.flops
    return perfect_time / compute_time

  def get_system_efficiency(self):
    return (self.get_bw_time() + self.get_fw_time()) / \
      self.get_total_time()

  def get_total_efficiency(self):
    total_flops = self.get_useful_flops()
    perfect_time = self._blocks_per_proc * self.exe._num_microbatches * \
      total_flops / self.sys.matrix.flops
    return perfect_time / self.get_total_time()

  def get_weight_space(self):
    return self._weight_space

  def get_act_space(self):
    return self._act_space

  def get_act_checkpoint_size(self):
    if self.exe.training:
      if self.exe.activation_recompute != 'full':
        return 0
      else:
        return self._act_checkpoint_size
    else:
      return 0

  def get_weight_grad_space(self):
    if self.exe.training:
      return self._weight_grad_space
    else:
      return 0

  def get_act_grad_space(self):
    if self.exe.training:
      return self._act_grad_space
    else:
      return 0

  def get_optimizer_space(self):
    if self.exe.training:
      return self._optimizer_space
    else:
      return 0

  def _get_mem_cap_reqs(self):
    tier1 = 0
    tier2 = 0
    if self.exe.weight_offload:
      tier1 += self._block_weight_space * 2
      tier2 += self.get_weight_space()
    else:
      tier1 += self.get_weight_space()
    if self.exe.activations_offload:
      tier1 += self._block_act_space * 2
      tier2 += self.get_act_space()
    else:
      tier1 += self.get_act_space()
    tier1 += self.get_act_checkpoint_size()
    if self.exe.optimizer_offload:
      # We keep one set of non-sharded weight grads after compute before
      # reduction, and one sharded set for offloading
      tier1 += self._block_weight_grad_space_no_sharding + \
        self._block_weight_grad_space
      tier2 += self._block_weight_grad_space * self._blocks_per_proc
      tier1 += self._block_optimizer_space * 2
      tier2 += self._optimizer_space
    else:
      tier1 += self.get_weight_grad_space() + \
        self.get_optimizer_space()
    tier1 += self.get_act_grad_space()
    return tier1, tier2

  def get_mem_tier1_cap_req(self):
    return self._get_mem_cap_reqs()[0]

  def get_mem_tier2_cap_req(self):
    return self._get_mem_cap_reqs()[1]

  def get_act_offload_bw_req(self):
    # We should be able to offload (write) activation during FW pass and
    # prefetch it (read) during BW pass for block (i-1)
    # After BW pass activations are discarded
    offload_time = min(
      self._baseblock_fw_time - self._block_fw_mem_time,
      self._edgeblock_fw_time - self._block_fw_mem_time)
    return self._block_act_space / offload_time

  def get_weight_offload_bw_req(self):
    # We should be able to offload (write) and prefetch (read) weights both
    # during FW and BW passes for blocks (i-1) / (i+1).
    # We always keep weights, they cannot be discarded
    offload_time = min(
      self._baseblock_fw_time - self._block_fw_mem_time,
      self._edgeblock_fw_time - self._block_fw_mem_time)
    return self._block_weight_space / offload_time

  def get_optim_offload_bw_req(self):
    # We should be able to offload (write) weight grads and optimizer state
    # and prefetch (read) optimizer state during BW passes for blocks
    # (i-1) / (i+1).
    if self.exe.training:
      offload_time = min(
        self._baseblock_bw_time - self._block_bw_mem_time,
        self._edgeblock_bw_time - self._block_bw_mem_time)
      return (self._block_weight_grad_space + self._block_optimizer_space) / \
        offload_time
    else:
      return 0

  def get_offload_mem_bw_req(self):
    fw_offload_time = min(
      self._baseblock_fw_time - self._block_fw_mem_time,
      self._edgeblock_fw_time - self._block_fw_mem_time)
    if self.exe.training:
      bw_offload_time = min(
        self._baseblock_bw_time - self._block_bw_mem_time,
        self._edgeblock_bw_time - self._block_bw_mem_time)
      req_bw = max(self._get_fw_offload_size() / fw_offload_time,
                   self._get_bw_offload_size() / bw_offload_time)
      return req_bw
    else:
      return self._get_fw_offload_size() / fw_offload_time

  def display_stats(self):
    stats = "=" * 80 + "\n"
    stats += "" \
      f"Model {self.app.name}: {self.app.num_blocks} blocks, " \
      f"hidden={self.app.hidden}, num attn heads: {self.app.attn_heads}\n" \
      f"Run on {self.exe.num_procs} processors with TP={self.exe.tensor_par}, PP={self.exe.pipeline_par}, " \
      f"DP={self.exe.data_par}, {self._blocks_per_proc} blocks per processor\n" \
      f"Execution: {self.exe.cfg};\n" \
      f"System: {self.sys.cfg};\n" \
      f"Weights: {human_format(self.get_weight_space(), 'bytes')};\n" \
      f"Act: {human_format(self.get_act_space(), 'bytes')};\n" \
      f"Act CP: {human_format(self.get_act_checkpoint_size(), 'bytes')};\n" \
      f"Act grad: {human_format(self.get_act_grad_space(), 'bytes')};\n" \
      f"Weight grad: {human_format(self.get_weight_grad_space(), 'bytes')};\n" \
      f"Optim space: {human_format(self.get_optimizer_space(), 'bytes')};\n" \
      f"Batch FW time: {self.get_fw_time():.4f};\n" \
      f"Batch BW time: {self.get_bw_time():.4f};\n" \
      f"Batch FW offload overhead: {self.get_fw_offload_overhead():.4f};\n" \
      f"Batch BW offload overhead: {self.get_bw_offload_overhead():.4f};\n" \
      f"Batch recompute overhead: {self.get_recompute_time():.4f};\n" \
      f"Batch recomm overhead: {self.get_recomm_time():.4f};\n" \
      f"Batch bubble overhead: {self.get_bubble_time():.4f};\n" \
      f"Batch TP comm time: {self.get_tp_comm_time():.4f};\n" \
      f"Batch PP comm time: {self.get_pp_comm_time():.4f};\n" \
      f"Batch DP comm overhead: {self.get_dp_comm_time():.4f};\n" \
      f"Batch total time: {self.get_total_time():.4f};\n" \
      f"Activation offload required BW: {human_format(self.get_act_offload_bw_req(), 'bandwidth')};\n" \
      f"Weight offload required BW: {human_format(self.get_weight_offload_bw_req(), 'bandwidth')};\n" \
      f"Optimizer offload required BW: {human_format(self.get_optim_offload_bw_req(), 'bandwidth')};\n" \
      f"Total offload required BW: {human_format(self.get_offload_mem_bw_req(), 'bandwidth')};\n" \
      f"Mem tier1 capacity requirement: {human_format(self.get_mem_tier1_cap_req(), 'bytes')};\n" \
      f"Mem tier2 capacity requirement: {human_format(self.get_mem_tier2_cap_req(), 'bytes')};\n" \
      f"Mem Tier2 BW for offload: {human_format(self.get_offload_mem_bw_req(), 'bandwidth')};\n" \
      f"Compute efficiency: {self.get_compute_efficiency()*100:.2f}%;\n" \
      f"System efficiency: {self.get_system_efficiency()*100:.2f}%;\n" \
      f"Total efficiency: {self.get_total_efficiency()*100:.2f}%;\n"
    self.log.info(stats)
