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

class Processor:
  """Configuration for a processing engine."""

  def __init__(self, cfg):
    self._datatypes = {}
    for datatype in cfg.keys():
      self._datatypes[datatype] = {
        'flops': cfg[datatype]['tflops'] * 1e12,
        'efficiency': []
      }
      last = None
      for gflops, eff in cfg[datatype]['gflops_efficiency']:
        flops = gflops * 1e9
        assert 0 < eff <= 1.0
        if last:
          assert flops < last
        last = flops
        self._datatypes[datatype]['efficiency'].append((flops, eff))

  def flops(self, datatype):
    return self._datatypes[datatype]['flops']

  def efficiency(self, datatype, op_flops):
    for flops, eff in self._datatypes[datatype]['efficiency']:
      if op_flops >= flops:
        return eff
    assert False, f'{op_flops} wasn\'t covered in {datatype} efficiency curve'

  def throughput(self, datatype, op_flops):
    assert datatype in self._datatypes, f'Unsupported type: {datatype}'
    return self.flops(datatype) * self.efficiency(datatype, op_flops)
