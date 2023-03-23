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


class Layer:
  """
  A single layer of a neural network. Has weights, activation space,
  gradients, and optimizer state associated with it. May invoke compute,
  memory access, or network operation.
  """

  def __init__(self, name, fw_flops=0, agrad_flops=0, wgrad_flops=0,
               inputs_size=0, output_size=0, activation_space=0,
               activation_grads=0, weight_space=0, weight_grads=0,
               optim_space=0, needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    self.name = name
    self.fw_flops = fw_flops
    self.agrad_flops = agrad_flops
    self.wgrad_flops = wgrad_flops
    self.inputs_size = inputs_size
    self.output_size = output_size
    # activations equal input size, we store them to compute Wgrad during BW
    self.activation_space = activation_space
    # activation grads equal output size and correspond grads w.r.t. the output
    self.activation_grads = activation_grads
    self.weight_space = weight_space
    self.weight_grads = weight_grads
    self.optim_space = optim_space
    self.optim_sharding_num_proc = 1

    # Add optimizations and parallelization split
    self.needs_recompute = needs_recompute
    self.activation_reused=activation_reused
    self.activation_stored = activation_stored
    self.output_stored = output_stored
    # Before bytes_per_element set by SW config, we operate with just
    # parameter count, setting bytes_per_element to 1
    self.bytes_per_element = 1
    self.processing_time = None

  def get_stats_json(self):
    return {
      'name': self.name,
      'fw_flops': self.get_fw_flops(),
      'fw_mem_accessed': self.get_fw_mem_accessed(),
      'fw_arithmetic_intensity': self.get_fw_arithmetic_intensity(),
      'agrad_flops': self.get_agrad_flops(),
      'agrad_mem_accessed': self.get_agrad_mem_accessed(),
      'agrad_arithmetic_intensity': self.get_agrad_arithmetic_intensity(),
      'wgrad_flops': self.get_wgrad_flops(),
      'wgrad_mem_accessed': self.get_wgrad_mem_accessed(),
      'wgrad_arithmetic_intensity': self.get_wgrad_arithmetic_intensity(),
      'optim_flops': self.get_optim_step_flops(),
      'optim_mem_accessed': self.get_optim_step_mem_accessed(),
      'optim_arithmetic_intensity': self.get_optim_step_arithmetic_intensity(),
      'weight': self.get_weight(),
      'activation': self.get_activation(),
      'weight_grad': self.get_weight_grad(),
      'activation_grad': self.get_activation_grad(),
      'optimizer': self.get_optimizer()
    }

  def get_stats_str(self):
    stats = "Operation {0}:\n{1} FW flops, {2} FW bytes accessed,".format(
      self.name,
      human_format(self.get_fw_flops(), 'flops'),
      human_format(self.get_fw_mem_accessed(), 'bytes'))
    stats += " FW AI: {0:.3f}\n".format(self.get_fw_arithmetic_intensity())
    stats += "{0} BW Adrad flops, {1} BW Agrad bytes accessed,".format(
      human_format(self.get_agrard_flops(), 'flops'),
      human_format(self.get_agrad_mem_accessed(), 'bytes'))
    stats += " BW Agrad AI: {0:.3f}\n".format(
      self.get_agrad_arithmetic_intensity())
    stats += "{0} BW Wdrad flops, {1} BW Wgrad bytes accessed,".format(
      human_format(self.get_wgrard_flops(), 'flops'),
      human_format(self.get_wgrad_mem_accessed(), 'bytes'))
    stats += " BW Wgrad AI: {0:.3f}\n".format(
      self.get_wgrad_arithmetic_intensity())
    stats += "{0} Optim flops, {1} Optim bytes accessed,".format(
      human_format(self.get_optim_step_flops(), 'flops'),
      human_format(self.get_optim_step_mem_accessed(), 'bytes'))
    stats += " Optim AI: {0:.3f}\n".format(
      self.get_optim_step_arithmetic_intensity())
    stats += "W: {0}, Act: {1}, WGrad: {2}, AGrad: {3}, Optim: {4}".format(
      human_format(self.get_weight(), 'bytes'),
      human_format(self.get_activation(), 'bytes'),
      human_format(self.get_weight_grad(), 'bytes'),
      human_format(self.get_activation_grad(), 'bytes'),
      human_format(self.get_optimizer(), 'bytes'))
    return stats

  def set_bytes_per_element(self, bytes_per_element):
    self.bytes_per_element = bytes_per_element

  # Shard (distribute) optimizer and weight grads between data parallel nodes
  def shard_optimizer(self, num_procs):
    self.optim_sharding_num_proc = num_procs

  # getters that will be called from Megatron model class, can be rewritten
  def get_fw_flops(self):
    return self.fw_flops

  def get_fw_mem_accessed(self):
    mem_accessed = self.inputs_size + self.output_size + self.weight_space
    mem_accessed *= self.bytes_per_element
    return mem_accessed

  def get_fw_arithmetic_intensity(self):
    if self.fw_flops == 0:
      return 0
    if self.get_fw_mem_accessed() == 0:
      return float('inf')
    return self.fw_flops / self.get_fw_mem_accessed()

  def get_recompute_flag(self):
    return self.needs_recompute

  def reuses_activation(self):
    return self.activation_reused

  def stores_activation(self):
    return self.activation_stored

  def stores_output(self):
    return self.output_stored

  def get_agrad_flops(self):
    return self.agrad_flops

  def get_agrad_mem_accessed(self):
    # activation grads equal output size and correspond grads w.r.t.
    # layer output; activations are equal to input size
    grad_mem = self.weight_space + (
      self.activation_space + self.activation_grads)
    grad_mem *= self.bytes_per_element
    return grad_mem

  def get_agrad_arithmetic_intensity(self):
    if self.agrad_flops == 0:
      return 0
    if self.get_agrad_mem_accessed() == 0:
      return float('inf')
    return self.agrad_flops / self.get_agrad_mem_accessed()

  def get_wgrad_flops(self):
    return self.wgrad_flops

  def get_wgrad_mem_accessed(self):
    if self.weight_space == 0:
      assert self.wgrad_flops == 0, \
        f"Haven't expected to see wgrad flops in layer {self.name}"
      return 0
    # activation grads equal output size and correspond grads w.r.t.
    # layer output; activations are equal to input size
    grad_mem = self.weight_grads + (
      self.activation_space + self.activation_grads)
    grad_mem *= self.bytes_per_element
    return grad_mem

  def get_wgrad_arithmetic_intensity(self):
    if self.wgrad_flops == 0:
      return 0
    if self.get_wgrad_mem_accessed() == 0:
      return float('inf')
    return self.wgrad_flops / self.get_wgrad_mem_accessed()

  # We use Adam optimizer. The amount of flops is based on the number of
  # weight grads to accommodate for possible weight_grad sharding
  # among data parallel nodes
  def get_optim_step_flops(self):
    optim_flops = self.weight_grads / self.optim_sharding_num_proc * 11
    return optim_flops

  def get_optim_step_mem_accessed(self):
    return self.get_optimizer()

  def get_optim_step_arithmetic_intensity(self):
    if self.get_optim_step_flops() == 0:
      return 0
    if self.get_optim_step_mem_accessed() == 0:
      return float('inf')
    return self.get_optim_step_flops() / self.get_optim_step_mem_accessed()

  def get_weight(self):
    return self.weight_space * self.bytes_per_element

  def get_activation(self):
    return self.activation_space * self.bytes_per_element

  def get_output(self):
    return self.output_size * self.bytes_per_element

  def get_weight_grad(self, sharded=True):
    # Keep lower precision copy of grads for mem and net transfers
    grads = self.weight_grads
    if sharded:
      # We keep grads in lower precision for communication
      grads *= self.bytes_per_element
      grads /= self.optim_sharding_num_proc
    else:
      # otherwise keep grads in 32 bit for accumulation
      grads *= 4
    return grads

  def get_activation_grad(self):
    return self.activation_grads * self.bytes_per_element

  def get_optimizer(self):
    # Keep 32-bits master copy of weights, plus both moments (m,v)
    # master copy for grads is accounted for in get_weight_grad()
    moments_size = self.optim_space * 4
    if self.bytes_per_element < 4:
      master_copy_size = self.weight_space * 4
    else:
      master_copy_size = 0
    return (master_copy_size + moments_size) / self.optim_sharding_num_proc

  def set_processing_time(self, processing_time):
    self.processing_time = processing_time

  def get_processing_time(self):
    return self.processing_time

  def use_matrix_engine(self):
    return False

  def compute_flops_time(self, sys, stage):
    if stage == "fw":
      flops = self.get_fw_flops()
    elif stage == "agrad":
      flops = self.get_agrad_flops()
    elif stage == "wgrad":
      flops = self.get_wgrad_flops()
    elif stage == "optim":
      flops = self.get_optim_step_flops()
    else:
      raise Exception(f'Bad compute stage : {stage}') 
    if self.use_matrix_engine() and stage != "optim":
      throughput = sys.get_matrix_throughput(flops)
    else:
      throughput = sys.get_vector_throughput(flops)
    return flops / throughput

  def compute_mem_time(self, sys, stage):
    if stage == "fw":
      mem = self.get_fw_mem_accessed()
    elif stage == "agrad":
      mem = self.get_agrad_mem_accessed()
    elif stage == "wgrad":
      mem = self.get_wgrad_mem_accessed()
    elif stage == "optim":
      mem = self.get_optim_step_mem_accessed()
    else:
      raise Exception(f'Bad compute stage : {stage}') 
    return mem / sys.get_mem1_throughput(mem)

  def compute_net_time(self, sys, stage):
    return 0

  def compute_processing_time(self, sys, stage):
    self.processing_time =  sys.get_processing_time(
      self.compute_flops_time(sys, stage),
      self.compute_mem_time(sys, stage)
    )
    return self.processing_time

# We can factor all layers peculiarities and layer-wise optimizations by
# rewriting parent class member functions when needed
class Linear(Layer):
  def __init__(self, name, batch_seq, c_in, c_out,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    m, n, k = batch_seq, c_in, c_out
    super().__init__(name,
                     fw_flops=2*m*n*k,
                     agrad_flops=2*m*n*k,
                     wgrad_flops=2*m*n*k,
                     inputs_size=m*n,
                     output_size=m*k,
                     weight_space=n*k,
                     weight_grads=n*k,
                     activation_space=m*n,
                     activation_grads=m*k,
                     optim_space=2*n*k,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    return True

class BatchMatMul(Layer):
  def __init__(self, name, batch, size_a, contraction_size, size_b,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    m, n, k = size_a, contraction_size, size_b
    super().__init__(name,
                     fw_flops=batch*2*m*n*k,
                     agrad_flops=batch*2*2*m*n*k,
                     inputs_size=batch*(m*n+n*k),
                     output_size=batch*m*k,
                     activation_space=batch*(m*n+n*k),
                     activation_grads=batch*m*k,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def use_matrix_engine(self):
    return True

# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# https://cthorey.github.io./blog/2016/backpropagation/
class LayerNorm(Layer):
  def __init__(self, name, act_size, hidden,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    super().__init__(name,
                     fw_flops=9*act_size,
                     agrad_flops=14*act_size,
                     wgrad_flops=7*act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     weight_space=2*hidden,
                     weight_grads=2*hidden,
                     optim_space=2*2*hidden,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)


class DropOut(Layer):
  def __init__(self, name, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    super().__init__(name,
                     fw_flops=act_size,
                     agrad_flops=act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)


  # need to account for DropOut mask of bool type that takes 1 B per element
  # mask is the only DropOut activation
  def get_activation(self):
    return self.activation_space

  def get_activation_grad(self):
    return self.activation_grads

  def get_fw_mem_accessed(self):
    mask_size = self.activation_space
    mem_accessed = self.inputs_size + self.output_size
    mem_accessed *= self.bytes_per_element
    mem_accessed += mask_size
    return mem_accessed

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://mlfromscratch.com/activation-functions-explained/#/
class GeLU(Layer):
  def __init__(self, name, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True,
               fused=False):
    # Fused GeLU runs right after previous Linear layer and does not store
    # activations or gradients
    self._fused = fused
    if fused:
      eff_act_space = 0
      eff_act_grads = 0
    else:
      eff_act_space = act_size
      eff_act_grads = act_size
    super().__init__(name, fw_flops=8*act_size, agrad_flops=13*act_size,
                     inputs_size=act_size, output_size=act_size,
                     activation_space=eff_act_space,
                     activation_grads=eff_act_grads,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
class SoftMax(Layer):
  def __init__(self, name, act_size,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    super().__init__(name,
                     fw_flops=5*act_size,
                     agrad_flops=8*act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_agrad_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://explained.ai/matrix-calculus/#sec:1.4.2
class ElementWise(Layer):
  def __init__(self, name, operand1, operand2,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    act_size = max(operand1, operand2)
    super().__init__(name,
                     fw_flops=act_size,
                     agrad_flops=(operand1+operand2),
                     inputs_size=(operand1+operand2),
                     output_size=act_size,
                     activation_space=(operand1+operand2),
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)


# Splits activation on the forward pass, sums gradients on the backward
class Fork(Layer):
  def __init__(self, name, act_size, num_users,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    self.num_users = num_users
    super().__init__(name,
                     inputs_size=act_size,
                     agrad_flops=num_users*act_size,
                     activation_space=act_size,
                     # Gradients from num_users accumulated in a single storage
                     # that's accounted in the other layers
                     # use 0 here to avoid double accounting
                     activation_grads=0,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_fw_mem_accessed(self):
    return 0

  def get_agrad_mem_accessed(self):
    return self.activation_space * self.bytes_per_element * (
      self.num_users + 1)


class TPComm(Layer):
  def __init__(self, name, act_size, comm_size,
               split_comm=False, conjugate=False,
               in_network_reduction=False,
               needs_recompute=False, activation_reused=False,
               activation_stored=True, output_stored=True):
    self.comm_size = comm_size
    self.split_comm = split_comm
    self.conjugate = conjugate
    if self.comm_size == 1:
      fw_flops = 0
      bw_flops = 0
      in_size = 0
      out_size = 0
    else:
      if not self.conjugate:
        # FW pass Identity/AllGather, BW pass AllReduce/ReduceScatter
        fw_flops = 0
        if not in_network_reduction:
          bw_flops = act_size * (self.comm_size - 1) / self.comm_size
        else:
          bw_flops = 0
        in_size = act_size
        out_size = act_size
      else:
        # Conjugate function is opposite
        if not in_network_reduction:
          fw_flops = act_size * (self.comm_size - 1) / self.comm_size
        else:
          fw_flops = 0
        bw_flops = 0
        in_size = act_size
        out_size = act_size

    super().__init__(name,
                     fw_flops=fw_flops,
                     agrad_flops=bw_flops,
                     inputs_size=in_size,
                     output_size=out_size,
                     activation_space=in_size,
                     activation_grads=out_size,
                     needs_recompute=needs_recompute,
                     activation_reused=activation_reused,
                     activation_stored=activation_stored,
                     output_stored=output_stored)

  def get_activation(self):
    if self.split_comm:
      return self.activation_space * self.bytes_per_element / self.comm_size
    else:
      if self.conjugate:
        return self.activation_space * self.bytes_per_element
      else:
        # Identity
        return 0

  def get_fw_mem_accessed(self):
    if not self.split_comm and not self.conjugate:
      # Identity
      return 0
    else:
      return super().get_fw_mem_accessed()

  def get_activation_grad(self):
    if self.split_comm:
      return self.activation_space * self.bytes_per_element / self.comm_size
    else:
      if not self.conjugate:
        return self.activation_grads * self.bytes_per_element
      else:
        # Identity
        return 0

  def get_agrad_mem_accessed(self):
    if not self.split_comm and self.conjugate:
      # Identity
      return 0
    else:
      return super().get_agrad_mem_accessed()