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

    """
    # bw = GB/s
    # cap = GB
    self.mem_tier1_bw = cfg['mem_tier1_bw'] * 1e9
    self.mem_tier1_cap = cfg['mem_tier1_cap'] * 1024**3
    self.mem_tier1_eff = cfg['mem_tier1_eff']
    assert 0 < self.mem_tier1_eff <= 1.0

    self.mem_tier2_bw = cfg['mem_tier2_bw'] * 1e9
    self.mem_tier2_cap = cfg['mem_tier2_cap'] * 1024**3
    self.mem_tier2_eff = cfg['mem_tier2_eff']
    assert 0 < self.mem_tier2_eff <= 1.0
    """

    self.proc_mode = cfg['processing_mode']
    assert self.proc_mode in ['roofline', 'no_overlap']

    self.net_tier1 = Network(cfg['net_tier1'])
    self.net_tier2 = Network(cfg['net_tier2'])

  """
  def memory_throughput(self, tier):
    if tier == 1:
      return self.mem_tier1_bw * self.mem_tier1_eff
    elif tier == 2:
      return self.mem_tier2_bw * self.mem_tier2_eff
    else:
      assert False
  """

  def get_network(self, tier):
    if tier == 1:
      return self.net_tier1
    elif tier == 2:
      return self.net_tier2
    else:
      assert False, f'Bad network tier ID: {tier}'

  def compute_flops_time(self, layer, bw):
    flops = layer.get_bw_flops() if bw else layer.get_fw_flops()
    if layer.use_matrix_engine():
      throughput = self.matrix.throughput(flops)
    else:
      throughput = self.vector.throughput(flops)
    return flops / throughput

  def compute_offload_time(self, size):
    return size / self.mem2.throughput(size)

  def compute_mem_time(self, layer, bw):
    mem = layer.get_bw_mem_accessed() if bw else layer.get_fw_mem_accessed()
    return mem / self.mem1.throughput(mem)

  def compute_processing_time(self, layer, bw):
    return self._compute_processing_time(
      self.compute_flops_time(layer, bw),
      self.compute_mem_time(layer, bw)
    )

  def _compute_processing_time(self, flops_time, mem_time):
    if self.proc_mode == 'roofline':
      return max(flops_time, mem_time)
    elif self.proc_mode == 'no_overlap':
      return flops_time + mem_time
