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
import gzip
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, np.bool_):
      return bool(obj)
    return super(NpEncoder, self).default(obj)

def is_json_extension(filename):
  return filename.endswith('.json') or filename.endswith('.json.gz')


def write_json_file(jdata, filename):
  assert is_json_extension(filename)
  opener = gzip.open if filename.endswith('.gz') else open
  indent = None if filename.endswith('.gz') else 2
  with opener(filename, 'wb') as fd:
    fd.write(bytes(json.dumps(jdata, indent=indent, cls=NpEncoder), 'utf-8'))


def read_json_file(filename):
  assert is_json_extension(filename)
  opener = gzip.open if filename.endswith('.gz') else open
  with opener(filename, 'rb') as fd:
    return json.loads(fd.read().decode('utf-8'))
