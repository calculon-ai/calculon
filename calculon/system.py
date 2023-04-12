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

  TypeSizes = {
    'float8'   : 1,
    'float16'  : 2,
    'float32'  : 4,
    'bfloat16' : 2
  }

  @staticmethod
  def supported_datatypes():
    return list(System.TypeSizes.keys())

  def __init__(self, cfg):
    self.cfg = cfg
    self.matrix = Processor(cfg['matrix'])
    self.vector = Processor(cfg['vector'])
    self.datatype = None

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

  def set_datatype(self, datatype):
    assert datatype in System.TypeSizes, f'Unsupported data type: {datatype}'
    self.datatype = datatype

  def get_matrix_throughput(self, flops):
    return self.matrix.throughput(self.datatype, flops)

  def get_vector_throughput(self, flops):
    return self.vector.throughput(self.datatype, flops)

  def get_mem1_throughput(self, size):
    return self.mem1.throughput(size)

  def get_mem2_throughput(self, size):
    return self.mem2.throughput(size)

  def compute_offload_time(self, size):
    return size / self.mem2.throughput(size)

  def get_processing_time(self, flops_time, mem_time):
    if self.proc_mode == 'roofline':
      return max(flops_time, mem_time)
    elif self.proc_mode == 'no_overlap':
      return flops_time + mem_time
