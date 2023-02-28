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

from .memory import *
from .network import *
from .processor import *

class System:
  """Configuration for a system."""

  def __init__(self, cfg):
    self.cfg = cfg
    self.matrix = Processor(cfg['matrix'])
    self.vector = Processor(cfg['vector'])

    self.mem1 = Memory(cfg['mem1'])
    self.mem2 = Memory(cfg['mem2'])

    self.proc_mode = cfg['processing_mode']
    assert self.proc_mode in ['roofline', 'no_overlap']

    self.networks = [Network(n) for n in cfg['networks']]

  @property
  def num_networks(self):
    return len(self.networks)

  def get_network(self, tier):
    assert tier < len(self.networks), f'Bad network tier ID: {tier}'
    return self.networks[tier]

  def compute_flops_time(self, layer, stage):
    if stage == "fw":
      flops = layer.get_fw_flops()
    elif stage == "bw":
      flops = layer.get_bw_flops()
    elif stage == "optim":
      flops = layer.get_optim_step_flops()
    else:
      raise Exception(f'Bad compute stage : {stage}') 
    if layer.use_matrix_engine() and stage != "optim":
      throughput = self.matrix.throughput(flops)
    else:
      throughput = self.vector.throughput(flops)
    return flops / throughput

  def compute_offload_time(self, size):
    return size / self.mem2.throughput(size)

  def compute_mem_time(self, layer, stage):
    if stage == "fw":
      mem = layer.get_fw_mem_accessed()
    elif stage == "bw":
      mem = layer.get_bw_mem_accessed()
    elif stage == "optim":
      mem = layer.get_optim_step_mem_accessed()
    else:
      raise Exception(f'Bad compute stage : {stage}') 
    return mem / self.mem1.throughput(mem)

  def compute_processing_time(self, layer, stage):
    return self._compute_processing_time(
      self.compute_flops_time(layer, stage),
      self.compute_mem_time(layer, stage)
    )

  def _compute_processing_time(self, flops_time, mem_time):
    if self.proc_mode == 'roofline':
      return max(flops_time, mem_time)
    elif self.proc_mode == 'no_overlap':
      return flops_time + mem_time
