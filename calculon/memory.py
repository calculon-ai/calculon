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

class Memory:
  """Configuration for a memory."""

  def __init__(self, cfg):
    self._capacity = cfg['GiB'] * 1024**3
    self._bandwidth = cfg['GBps'] * 1e9
    self._efficiency = []
    for mbytes, eff in cfg['MB_efficiency']:
      bytes = mbytes * 1e6
      assert 0 < eff <= 1.0
      self._efficiency.append((bytes, eff))

  @property
  def capacity(self):
    return self._capacity

  @property
  def bandwidth(self):
    return self._bandwidth

  def efficiency(self, op_bytes):
    for bytes, eff in self._efficiency:
      if op_bytes >= bytes:
        return eff
    assert False, f'OP bytes {op_bytes} wasn\'t covered'

  def throughput(self, op_bytes):
    return self._bandwidth * self.efficiency(op_bytes)
