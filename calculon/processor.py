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
    self._flops = cfg['tflops'] * 1e12
    self._efficiency = []
    for gflops, eff in cfg['gflops_efficiency']:
      flops = gflops * 1e9
      assert 0 < eff <= 1.0
      self._efficiency.append((flops, eff))

  @property
  def flops(self):
    return self._flops

  def efficiency(self, op_flops):
    for flops, eff in self._efficiency:
      if op_flops >= flops:
        return eff
    assert False, f'OP flops {op_flops} wasn\'t covered'

  def throughput(self, op_flops):
    return self._flops * self.efficiency(op_flops)
