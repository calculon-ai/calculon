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

  def __init__(self, name, fw_flops=0, bw_flops=0, inputs_size=0,
               output_size=0, activation_space=0, activation_grads=0,
               weight_space=0, weight_grads=0, optim_space=0,
               #net_access_rd=0, net_access_wr=0,
               needs_recompute=False, activation_reuse=False):
    self.name = name
    self.fw_flops = fw_flops
    self.bw_flops = bw_flops
    self.inputs_size = inputs_size
    self.output_size = output_size
    self.activation_space = activation_space
    self.activation_grads = activation_grads
    self.weight_space = weight_space
    self.weight_grads = weight_grads
    self.optim_space = optim_space

    # TODO: do we need these two rows?
    # self.net_access_rd = net_access_rd
    # self.net_access_wr = net_access_wr

    # HW parameters set during the run with HW config
    # self.compute_throughput = float('inf')
    # self.mem_throughput = float('inf')
    # self.mem_offload_throughput = float('inf')
    # self.net_throughput = float('inf')

    # Add optimizations and parallelization split
    self.needs_recompute = needs_recompute
    self.activation_reuse = activation_reuse
    # Before bytes_per_element set by SW config, we operate with just
    # parameter count, setting bytes_per_element to 1
    self.bytes_per_element = 1

  def display_stats(self):
    stats = "Operation {0}:\n{1} FW flops, {2} FW bytes accessed,".format(
      self.name,
      human_format(self.get_fw_flops(), 'flops'),
      human_format(self.get_fw_mem_accessed(), 'bytes'))
    stats += " FW AI: {0:.3f}\n".format(self.get_fw_arithmetic_intensity())
    stats += "{0} BW flops, {1} BW bytes accessed,".format(
      human_format(self.get_bw_flops(), 'flops'),
      human_format(self.get_bw_mem_accessed(), 'bytes'))
    stats += " BW AI: {0:.3f}\n".format(self.get_bw_arithmetic_intensity())
    stats += "W: {0}, Act: {1}, WGrad: {2}, AGrad: {3}, Optim: {4}".format(
      human_format(self.get_weight(), 'bytes'),
      human_format(self.get_activation(), 'bytes'),
      human_format(self.get_weight_grad(), 'bytes'),
      human_format(self.get_activation_grad(), 'bytes'),
      human_format(self.get_optim(), 'bytes'))
    print(stats)

  def set_bytes_per_element(self, bytes_per_element):
    self.bytes_per_element = bytes_per_element

  # Shard (distribute) optimizer and weight grads between data parallel nodes
  def shard_optimizer(self, num_gpus):
    self.weight_grads /= num_gpus
    self.optim_space /= num_gpus

  # setters related to HW config
  # TODO get rid of them as we probably need only flops and bytes from layer
  # def set_compute_throughput(self, compute_throughput):
  #   self.compute_throughput = compute_throughput

  # def set_mem_throughput(self, mem_throughput):
  #   self.mem_throughput = mem_throughput

  # TODO get rid of it as mem offload assessed on App (Megatron) level
  # def set_mem_offload_throughput(self, mem_offload_throughput):
  #   self.mem_offload_throughput = mem_offload_throughput

  # TODO get rid of it as network assessed on App (Megatron) level
  # def set_net_throughput(self, net_throughput):
  #   self.net_throughput = net_throughput

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

  # TODO get rid of it as time is assessed on App (Megatron) level
  # def get_fw_time(self):
  #     compute_time = self.fw_flops / self.compute_throughput
  #     mem_time =  self.get_fw_mem_accessed() / self.mem_throughput
  #     return compute_time + mem_time

  # We add optimizer (Adam) computation to BW grads. The amount of flops is
  # based on number of weight grads to accommodate for possible weight_grad
  # sharding among data parallel nodes
  def get_bw_flops(self):
    optim_flops = self.weight_grads * 11
    return self.bw_flops + optim_flops

  def get_bw_mem_accessed(self):
    fw_mem = self.get_fw_mem_accessed()
    # activation grads equal input size and correspond to computed
    # activation grads, input grads are equal to outtput size
    grad_mem = self.weight_grads + self.activation_grads + self.output_size
    grad_mem *= self.bytes_per_element
    return fw_mem + grad_mem + self.get_optim()

  def get_bw_arithmetic_intensity(self):
    if self.bw_flops == 0:
      return 0
    if self.get_bw_mem_accessed() == 0:
      return float('inf')
    return self.bw_flops / self.get_bw_mem_accessed()

  # TODO get rid of it as time is assessed on App (Megatron) level
  # def get_bw_time(self):
  #     compute_time = self.bw_flops / self.compute_throughput
  #     mem_time =  self.get_bw_mem_accessed() / self.mem_throughput
  #     return compute_time + mem_time

  def get_weight(self):
    return self.weight_space * self.bytes_per_element

  def get_activation(self):
    if self.activation_reuse:
      return 0
    return self.activation_space * self.bytes_per_element

  def get_weight_grad(self):
    return self.weight_grads * self.bytes_per_element

  def get_activation_grad(self):
    return self.activation_grads * self.bytes_per_element

  def get_optim(self):
    moments_size = self.optim_space * 4
    # Keep 32-bits master copy of weights and grads, plus both moments (m,v)
    if self.bytes_per_element < 4:
      master_copy_size = (self.weight_grads + self.weight_space) * 4
    else:
      master_copy_size = 0
    return master_copy_size + moments_size

  # def get_flops(self):
  #     if self.sw_config['training']:
  #         if self.needs_recompute:
  #             return 2 * self.get_fw_flops() + self.get_bw_flops()
  #         else:
  #             return self.get_fw_flops() + self.get_bw_flops()
  #     else:
  #         return self.get_fw_flops()


# We can factor all layers peculiarities and layer-wise opttimizations by
# rewriting parent class member functions when needed
class Linear(Layer):
  def __init__(self, name, batch_seq, c_in, c_out,
               needs_recompute=False, activation_reuse=False):
    m, n, k = batch_seq, c_in, c_out
    super().__init__(name,
                     fw_flops=2*m*n*k,
                     bw_flops=2*2*m*n*k,
                     inputs_size=m*n,
                     output_size=m*k,
                     weight_space=n*k,
                     weight_grads=n*k,
                     activation_space=m*k,
                     activation_grads=m*n,
                     optim_space=2*n*k,
                     needs_recompute=needs_recompute,
                     activation_reuse=activation_reuse)


class BatchMatMul(Layer):
  def __init__(self, name, batch, size_a, contraction_size, size_b,
               needs_recompute=False, activation_reuse=False):
    m, n, k = size_a, contraction_size, size_b
    super().__init__(name,
                     fw_flops=batch*2*m*n*k,
                     bw_flops=batch*2*2*m*n*k,
                     inputs_size=batch*(m*n+n*k),
                     output_size=batch*m*k,
                     activation_space=batch*m*k,
                     activation_grads=batch*(m*n+n*k),
                     needs_recompute=needs_recompute,
                     activation_reuse=activation_reuse)


# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# https://cthorey.github.io./blog/2016/backpropagation/
class LayerNorm(Layer):
  def __init__(self, name, act_size, hidden,
               needs_recompute=False, activation_reuse=False):
    super().__init__(name,
                     fw_flops=9*act_size,
                     bw_flops=21*act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     weight_space=2*hidden,
                     weight_grads=2*hidden,
                     optim_space=2*2*hidden,
                     needs_recompute=needs_recompute,
                     activation_reuse=activation_reuse)


class DropOut(Layer):
  def __init__(self, name, act_size,
               needs_recompute=False, activation_reuse=False):
    super().__init__(name,
                     fw_flops=act_size,
                     bw_flops=act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reuse=activation_reuse)


  # need to account for DropOut mask of bool type that takes 1 B per element
  # mask is the only DropOut activation
  def get_activation(self):
    if self.activation_reuse:
      return 0
    return self.activation_space

  def get_activation_grad(self):
    return self.activation_grads

  def get_fw_mem_accessed(self):
    mask_size = self.activation_space
    mem_accessed = self.inputs_size + self.output_size
    mem_accessed *= self.bytes_per_element
    mem_accessed += mask_size
    return mem_accessed

  def get_bw_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://mlfromscratch.com/activation-functions-explained/#/
class GeLU(Layer):
  def __init__(self, name, act_size,
               needs_recompute=False, activation_reuse=False):
    super().__init__(name, fw_flops=8*act_size, bw_flops=13*act_size,
                     inputs_size=act_size, output_size=act_size,
                     activation_space=act_size, activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reuse=activation_reuse)

  def get_bw_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
class SoftMax(Layer):
  def __init__(self, name, act_size,
               needs_recompute=False, activation_reuse=False):
    super().__init__(name,
                     fw_flops=5*act_size,
                     bw_flops=8*act_size,
                     inputs_size=act_size,
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=act_size,
                     needs_recompute=needs_recompute,
                     activation_reuse=activation_reuse)

  def get_bw_mem_accessed(self):
    return self.get_fw_mem_accessed()


# https://explained.ai/matrix-calculus/#sec:1.4.2
class ElementWise(Layer):
  def __init__(self, name, operand1, operand2,
               needs_recompute=False, activation_reuse=False):
    act_size = max(operand1, operand2)
    super().__init__(name,
                     fw_flops=act_size,
                     bw_flops=(operand1+operand2),
                     inputs_size=(operand1+operand2),
                     output_size=act_size,
                     activation_space=act_size,
                     activation_grads=(operand1+operand2),
                     needs_recompute=needs_recompute,
                     activation_reuse=activation_reuse)


# Splits activation on the forward pass, sums gradients on the backward
class Fork(Layer):
  def __init__(self, name, act_size, num_users,
               needs_recompute=False, activation_reuse=False):
    self.num_users = num_users
    super().__init__(name,
                     fw_flops=0,
                     bw_flops=num_users*act_size,
                     activation_space=0,
                     # Gradients from num_users accumulated in a single storage
                     # that's accounted in the previous layer (TPComm)
                     activation_grads=(1-num_users)*act_size,
                     needs_recompute=needs_recompute,
                     activation_reuse=activation_reuse)

  def get_bw_mem_accessed(self):
    return self.activation_space * self.bytes_per_element * (self.num_users + 1)


class TPComm(Layer):
  def __init__(self, name, act_size, comm_size,
               split_comm=False, conjugate=False):
    # FW pass Identity/AllGather, BW pass AllReduce/ReduceScatter
    fw_flops = 0
    bw_flops = act_size * (comm_size - 1)
    in_size = act_size
    out_size = act_size
    if split_comm and not conjugate:
      in_size /= comm_size
    # Conjugate function is opposite
    if conjugate:
      fw_flops = act_size * (comm_size - 1)
      bw_flops = 0
      if split_comm:
        out_size /= comm_size
    super().__init__(name,
                     fw_flops=fw_flops,
                     bw_flops=bw_flops,
                     inputs_size=in_size,
                     output_size=out_size,
                     activation_space=out_size,
                     activation_grads=in_size)
